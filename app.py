import os
import re
import emoji
import awsgi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import tensorflow.lite as tflite
from googletrans import Translator
from transformers import AutoTokenizer
from googleapiclient.discovery import build
from flask import Flask, request, render_template

app = Flask(__name__)
load_dotenv()
api_keys = os.getenv("YOUTUBE_API_KEYS").split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tokenizer_path = 'model/saved_tokenizer'
model_path = 'model/model_float16.tflite'

fine_tuned_tokenizer = None
interpreter = None
input_details = None
output_details = None
SENTIMENT_LABELS = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Annoyed', 4: 'Fear', 5: 'Surprise'}

def load_model_and_tokenizer():
    global fine_tuned_tokenizer, interpreter, input_details, output_details
    if fine_tuned_tokenizer is None or interpreter is None or input_details is None or output_details is None:
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

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

def remove_emojis(text):
  text = emoji.replace_emoji(text, replace="")
  if text =='':
    return 'This is a empty comment'
  else : 
    return text

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
    
    # Step 2: Categorize by word count ranges
    def categorize_by_length(comments):
        priority_1 = []  # 20-30 words
        priority_2 = []  # 30-40 words
        priority_3 = []  # 0-20 words
        priority_4 = []  # Rest
        
        for comment in comments:
            word_count = len(comment.split())
            if 20 <= word_count <= 30:
                priority_1.append(comment)
            elif 30 < word_count <= 40:
                priority_2.append(comment)
            elif word_count < 20:
                priority_3.append(comment)
            else:
                priority_4.append(comment)
        
        # Concatenate all priority lists into one, respecting the order
        return priority_1 + priority_2 + priority_3 + priority_4

    other_language_comments = categorize_by_length(other_language_comments)
    hindi_comments = categorize_by_length(hindi_comments)

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

def tflite_predict(text):
    # Tokenize input text
    inputs = fine_tuned_tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=55)
    
    # Prepare input data for the TFLite model
    input_ids = inputs["input_ids"].flatten()
    attention_mask = inputs["attention_mask"].flatten()
    token_type_ids = inputs["token_type_ids"].flatten()
    
    # Ensure inputs are reshaped to match expected input shapes (batch size 1)
    input_ids = np.expand_dims(input_ids, axis=0).astype(np.int64)           # Reshape to (1, 55) and cast to INT64
    attention_mask = np.expand_dims(attention_mask, axis=0).astype(np.int64) # Reshape to (1, 55) and cast to INT64
    token_type_ids = np.expand_dims(token_type_ids, axis=0).astype(np.int64) # Reshape to (1, 55) and cast to INT64
    
    # Set the input tensors
    interpreter.set_tensor(input_details[1]['index'], input_ids)
    interpreter.set_tensor(input_details[0]['index'], attention_mask)
    interpreter.set_tensor(input_details[2]['index'], token_type_ids)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output and process it
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output, axis=1)[0]
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
    load_model_and_tokenizer()
    for index, row in pre_processed_comments.iterrows():
        sentiment_index = tflite_predict(row['Comment'])  
        sentiment_label = SENTIMENT_LABELS[sentiment_index] 
        sentiment_counts[sentiment_label]+=1
        comments_by_sentiment[sentiment_label].append(row['Comment'])
    return (sentiment_counts,comments_by_sentiment)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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

# if __name__ == "__main__":
#     app.run(debug=True)

def handler(event, context):
    return awsgi.response(app, event, context)