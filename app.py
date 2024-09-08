import os
import re
import cld3
import time
import httpx
import emoji
import random
import numpy as np
import pandas as pd
import tensorflow.lite as tflite
from googletrans import Translator
from transformers import AutoTokenizer
from googleapiclient.discovery import build
from flask import Flask, request, render_template

app = Flask(__name__)
api_keys = os.getenv("YOUTUBE_API_KEYS").split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load tokenizer and TFLite interpreter once at the start
tokenizer_path = 'model/saved_tokenizer'
model_path = 'model/model_float16.tflite'
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

BATCH_SIZE = 10
SENTIMENT_LABELS = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Annoyed', 4: 'Fear', 5: 'Surprise'}

def build_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def get_comments(video_id, max_results=1000):
    comments = []
    current_index = 0
    youtube = build_youtube_client(api_keys[current_index])
    next_page_token = None

    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception:
            current_index = (current_index + 1) % len(api_keys)
            youtube = build_youtube_client(api_keys[current_index])
            break

    return pd.DataFrame(comments, columns=["Comment"])

def extract_youtube_video_id(url):
    short_link_pattern = r"(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]+)"
    long_link_pattern = r"(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)"

    short_link_match = re.match(short_link_pattern, url)
    long_link_match = re.match(long_link_pattern, url)

    if short_link_match:
        return short_link_match.group(3)
    elif long_link_match:
        return long_link_match.group(3)
    return ""

def trim_whitespace(s):
    return s.strip()

def remove_emojis(text):
    text = emoji.replace_emoji(text, replace="")
    return 'This is an empty comment' if text == '' else text

def detect_and_translate(comments: pd.DataFrame, required_count=10):
    translator = Translator()
    translated_comments = []
    hindi_comments = []
    other_language_comments = []
    english_comments = []

    def handle_rate_limit(attempt):
        delay = (2 ** attempt) + random.uniform(0, 1)
        print(f"Rate limit hit. Retrying after {delay:.2f} seconds...")
        time.sleep(delay)

    # Categorize comments by language
    for index, row in comments.iterrows():
        comment = row['Comment']
        try:
            detection = cld3.get_language(comment)
            if detection and detection.language == 'en':
                english_comments.append(comment)
            elif detection and (detection.language == 'hi' or detection.language == 'hi-Latn'):
                hindi_comments.append(comment)
            else:
                other_language_comments.append(comment)
        except Exception as e:
            print(f"Language detection error for comment at index {index}: {e}")
            other_language_comments.append('This is a neutral text')

    # Prioritize comments based on the specified order
    english_long_comments = sorted([c for c in english_comments if len(c.split()) > 5], key=lambda x: len(x.split()), reverse=True)
    english_short_comments = sorted([c for c in english_comments if len(c.split()) <= 5], key=lambda x: len(x.split()), reverse=True)
    other_short_comments = sorted([c for c in other_language_comments if len(c.split()) <= 5], key=lambda x: len(x.split()))
    other_long_comments = sorted([c for c in other_language_comments if len(c.split()) > 5], key=lambda x: len(x.split()))
    hindi_short_comments = sorted([c for c in hindi_comments if len(c.split()) <= 5], key=lambda x: len(x.split()))
    hindi_long_comments = sorted([c for c in hindi_comments if len(c.split()) > 5], key=lambda x: len(x.split()))

    # Fill translated_comments based on the priority order
    prioritized_comments = (
        english_long_comments + 
        english_short_comments + 
        other_short_comments + 
        other_long_comments + 
        hindi_short_comments + 
        hindi_long_comments
    )

    # Add high-priority English comments directly without translation
    while len(translated_comments) < required_count and prioritized_comments:
        next_comment = prioritized_comments.pop(0)
        if next_comment in english_comments:
            translated_comments.append(next_comment)
        else:
            break  # Break to start translating other languages

    # Collect remaining comments for translation, up to the required count
    comments_to_translate = prioritized_comments[:max(0, required_count - len(translated_comments))]

    # Batch translate the selected comments
    if comments_to_translate:
        try:
            translations = []
            for comment in comments_to_translate:
                for attempt in range(3):
                    try:
                        translation = translator.translate(comment, dest='en')
                        translations.append(translation.text if translation and translation.text else 'This is a neutral text')
                        break
                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        print(f"Translation retry {attempt + 1} failed: {e}")
                        time.sleep(1)
                    except Exception as e:
                        handle_rate_limit(attempt)
                else:
                    translations.append('This is a neutral text')  # Fallback if all retries fail
            translated_comments.extend(translations)
        except Exception as e:
            print(f"Batch translation error: {e}")
            translated_comments.extend(['This is a neutral text'] * len(comments_to_translate))

    return pd.DataFrame(translated_comments[:required_count], columns=['Comment'])

def tflite_predict_batch(text_batch):
    # Tokenize the entire batch at once
    inputs = fine_tuned_tokenizer(text_batch, return_tensors="np", padding="max_length", truncation=True, max_length=55)
    
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    token_type_ids = inputs["token_type_ids"].astype(np.int64)

    results = []
    # Reuse tensors in the batch inference process
    interpreter.resize_tensor_input(input_details[1]['index'], input_ids.shape)
    interpreter.resize_tensor_input(input_details[0]['index'], attention_mask.shape)
    interpreter.resize_tensor_input(input_details[2]['index'], token_type_ids.shape)
    interpreter.allocate_tensors()
    
    interpreter.set_tensor(input_details[1]['index'], input_ids)
    interpreter.set_tensor(input_details[0]['index'], attention_mask)
    interpreter.set_tensor(input_details[2]['index'], token_type_ids)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    results.extend(np.argmax(output, axis=1))

    return results

def predict_sentiment(dataframe):
    comments = dataframe['Comment'].tolist()
    predictions = []

    # Process comments in batches
    batches = [comments[i:i + BATCH_SIZE] for i in range(0, len(comments), BATCH_SIZE)]
    for batch in batches:
        predictions.extend(tflite_predict_batch(batch))

    return predictions

def text_pre_processing(translated_comment: pd.DataFrame) -> pd.DataFrame:
    # Only remove emojis, avoid lowercasing
    translated_comment['Comment'] = translated_comment['Comment'].apply(remove_emojis)
    return translated_comment

def get_sentiment(comments_df: pd.DataFrame, comment_count=10) -> tuple:
    sentiment_counts = {label: 0 for label in SENTIMENT_LABELS.values()}
    comments_by_sentiment = {label: [] for label in SENTIMENT_LABELS.values()}
    
    # Detect language, translate comments, and preprocess text
    translated_comment = detect_and_translate(comments_df, comment_count)
    pre_processed_comments = text_pre_processing(translated_comment)
    
    # Predict sentiments for the pre-processed comments
    sentiment_indices = predict_sentiment(pre_processed_comments)
    
    # Organize results by sentiment
    for index, row in pre_processed_comments.iterrows():
        sentiment_label = SENTIMENT_LABELS[sentiment_indices[index]]
        sentiment_counts[sentiment_label] += 1
        comments_by_sentiment[sentiment_label].append(row['Comment'])

    return sentiment_counts, comments_by_sentiment

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = trim_whitespace(request.form.get('video_url'))
        video_id = extract_youtube_video_id(video_url)
        comment_count = int(request.form.get('comment_count', 10))
        if video_id:
            # comment_to_fetch = comment_count * 10 if comment_count <= 30 else comment_count
            comment_to_fetch = comment_count * 10
            comments_df = get_comments(video_id, max_results=comment_to_fetch)
            if not comments_df.empty:
                sentiment_counts, comments_by_sentiment = get_sentiment(comments_df, comment_count)
                top_comment = comments_df.iloc[0]["Comment"]
                return render_template('index.html', 
                                       comment=top_comment, 
                                       sentiment_counts=sentiment_counts, 
                                       comments_by_sentiment=comments_by_sentiment)
            else:
                return render_template('index.html', error="No comments found for this video.")
        else:
            return render_template('index.html', error="Invalid link, please try another link.")
    
    return render_template('index.html', sentiment_counts={}, comments_by_sentiment={})

if __name__ == "__main__":
    app.run(debug=True)