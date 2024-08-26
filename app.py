import os
import re
import awsgi
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from googletrans import Translator
from googleapiclient.discovery import build
from transformers import AutoTokenizer
from flask import Flask, render_template, request

app = Flask(__name__)

load_dotenv()
api_keys = os.getenv("YOUTUBE_API_KEYS").split(',')

tokenizer_path = 'model/saved_tokenizer'
model_path = 'model/transformer'

fine_tuned_tokenizer = None
fine_tuned_model = None

def load_model_and_tokenizer():
    global fine_tuned_tokenizer, fine_tuned_model
    if fine_tuned_tokenizer is None or fine_tuned_model is None:
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        fine_tuned_model = tf.saved_model.load(model_path)
        print("model is loaded successfully")

SENTIMENT_LABELS = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Annoyed', 4: 'Fear', 5: 'Surprise'}

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

# Get the Sentiments form the text

def detect_and_translate(text):
    translator = Translator()
    try:
        detection = translator.detect(text)
        
        if detection.lang != 'en':
            translation = translator.translate(text, dest='en')
            if translation and translation.text:
                return translation.text
            else:
                return 'This is a neutral text'
        else:
            return text
    except Exception as e:
        print(f"Translation error: {e}")
        return 'This is a neutral text' 

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

def get_sentiment(comments_df : pd.DataFrame) -> tuple:
    sentiment_counts = {'Sadness': 0, 'Joy': 0, 'Love': 0, 'Annoyed': 0, 'Fear': 0, 'Surprise': 0}
    comments_by_sentiment = {'Sadness': [], 'Joy': [], 'Love': [], 'Annoyed': [], 'Fear': [], 'Surprise': []}
    for index, row in comments_df.iterrows():
        translated_comment = detect_and_translate(row['Comment'])
        sentiment_index = return_sentiment(translated_comment)  
        sentiment_label = SENTIMENT_LABELS[sentiment_index] 
        sentiment_counts[sentiment_label]+=1
        comments_by_sentiment[sentiment_label].append(translated_comment)
    return (sentiment_counts,comments_by_sentiment)

# App Endpoints

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        load_model_and_tokenizer()
        video_url = trim_whitespace(request.form.get('video_url'))
        print(f"url link is :-",video_url)
        video_id = extract_youtube_video_id(video_url)
        print(f"video_id is :-",video_id)
        comment_count = int(request.form.get('comment_count', 10))
        print(f"comment_count is :-",comment_count)
        if video_id:
            comments_df = get_comments(video_id,max_results=comment_count*2)
            comments_df.to_csv('comments_df.csv')
            if not comments_df.empty:
                output = get_sentiment(comments_df)
                sentiment_counts = output[0]
                comments_by_sentiment = output[1]
                print(f"sentiment_counts is :-",sentiment_counts)
                print(f"comments_by_sentiment is :-",comments_by_sentiment)
                top_comment = comments_df.iloc[0]["Comment"]
                print(f"top_comment link is :-",top_comment)
                return render_template('index.html', comment=top_comment,sentiment_counts=sentiment_counts, comments_by_sentiment=comments_by_sentiment)
            else:
                return render_template('index.html', error="No comments found for this video.")
        else:
            return render_template('index.html', error="Invalid link, please try another link.")
    
    return render_template('index.html', sentiment_counts={}, comments_by_sentiment={})

if __name__ == "__main__":
    app.run(debug=True)