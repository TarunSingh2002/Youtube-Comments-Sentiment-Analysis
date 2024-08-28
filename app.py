import os
import re
import emoji
import awsgi
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from googletrans import Translator
from transformers import AutoTokenizer
from googleapiclient.discovery import build
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
load_dotenv()
api_keys = os.getenv("YOUTUBE_API_KEYS").split(',')

tokenizer_path = 'model/saved_tokenizer'
model_path = 'model/transformer'

fine_tuned_tokenizer = None
fine_tuned_model = None
SENTIMENT_LABELS = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Annoyed', 4: 'Fear', 5: 'Surprise'}

# load model and tokenizer
def load_model_and_tokenizer():
    global fine_tuned_tokenizer, fine_tuned_model
    if fine_tuned_tokenizer is None or fine_tuned_model is None:
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        fine_tuned_model = tf.saved_model.load(model_path)
        # print("model is loaded successfully")

#  Get the comments form video
def build_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def rotate_api_key(current_index):
    return (current_index + 1) % len(api_keys)

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

        except Exception as e:
            print(f"Error: {e}")
            current_index = rotate_api_key(current_index)
            youtube = build_youtube_client(api_keys[current_index])
            break
    df = pd.DataFrame(comments, columns=["Comment"])
    return df

def extract_youtube_video_id(url):
    
    short_link_pattern = r"(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]+)"
    long_link_pattern = r"(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)"

    short_link_match = re.match(short_link_pattern, url)
    long_link_match = re.match(long_link_pattern, url)

    if short_link_match:
        video_id = short_link_match.group(3)
    elif long_link_match:
        video_id = long_link_match.group(3)
    else:
        return ""
    
    return video_id

def trim_whitespace(s):
    return s.strip()

# Text Pre Processing
def remove_emojis(text):
  return emoji.replace_emoji(text, replace="")

# Get the Sentiments form the text
def detect_and_translate(comments: pd.DataFrame, required_count=10):
    translator = Translator()
    translated_comments = []
    hindi_comments = []
    other_language_comments = []

    # Step 1: Detect language and categorize comments
    for index, row in comments.iterrows():
        comment = row['Comment']
        try:
            detection = translator.detect(comment)
            if detection.lang == 'en':
                other_language_comments.append(comment)
            elif detection.lang == 'hi':
                hindi_comments.append(comment)
            else:
                translation = translator.translate(comment, dest='en')
                if translation and translation.text:
                    other_language_comments.append(translation.text)
                else:
                    other_language_comments.append('This is a neutral text')
        except Exception as e:
            print(f"Translation error for comment at index {index}: {e}")
            other_language_comments.append('This is a neutral text')
    
    # Step 2: Sort comments by word length
    other_language_comments.sort(key=lambda x: len(x.split()), reverse=True)
    hindi_comments.sort(key=lambda x: len(x.split()), reverse=True)
    
    # Step 3: Collect the required number of comments
    while len(translated_comments) < required_count and other_language_comments:
        translated_comments.append(other_language_comments.pop(0))
    
    while len(translated_comments) < required_count and hindi_comments:
        try:
            translation = translator.translate(hindi_comments.pop(0), dest='en')
            if translation and translation.text:
                translated_comments.append(translation.text)
            else:
                translated_comments.append('This is a neutral text')
        except Exception as e:
            print(f"Translation error: {e}")
            translated_comments.append('This is a neutral text')

    # Step 4: Return the result as a DataFrame
    result_df = pd.DataFrame(translated_comments, columns=['Comment'])
    return result_df

def return_sentiment(text : str) -> np.int64 :
    inputs = fine_tuned_tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=55)
    input_dict = {
        'input_ids': tf.cast(inputs['input_ids'], tf.int64),
        'attention_mask': tf.cast(inputs['attention_mask'], tf.int64),
        'token_type_ids': tf.cast(inputs['token_type_ids'], tf.int64)
        }
    outputs = fine_tuned_model(input_dict)
    probabilities = tf.nn.softmax(outputs, axis=-1) 
    probabilities_np = probabilities.numpy()[0] 
    predicted_index = np.argmax(probabilities_np)
    return predicted_index

def text_pre_processing(translated_comment: pd.DataFrame) -> pd.DataFrame:
    # lower casing
    translated_comment['Comment']=translated_comment['Comment'].str.lower()
    # remove emoji
    translated_comment['Comment']=translated_comment['Comment'].apply(remove_emojis)
    return translated_comment

def get_sentiment(comments_df : pd.DataFrame,comment_count=10) -> tuple:
    sentiment_counts = {'Sadness': 0, 'Joy': 0, 'Love': 0, 'Annoyed': 0, 'Fear': 0, 'Surprise': 0}
    comments_by_sentiment = {'Sadness': [], 'Joy': [], 'Love': [], 'Annoyed': [], 'Fear': [], 'Surprise': []}
    translated_comment = detect_and_translate(comments_df,comment_count)
    pre_processed_comments = text_pre_processing(translated_comment)
    for index, row in pre_processed_comments.iterrows():
        sentiment_index = return_sentiment(row['Comment'])  
        sentiment_label = SENTIMENT_LABELS[sentiment_index] 
        sentiment_counts[sentiment_label]+=1
        comments_by_sentiment[sentiment_label].append(row['Comment'])
    return (sentiment_counts,comments_by_sentiment)

# Flask App
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        load_model_and_tokenizer()
        video_url = trim_whitespace(request.form.get('video_url'))
        video_id = extract_youtube_video_id(video_url)
        comment_count = int(request.form.get('comment_count', 10))

        if video_id:
            comments_df = get_comments(video_id, max_results=comment_count * 10)
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