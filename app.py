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
from flask import Flask, render_template, request, render_template_string

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

class_labels = ['Sadness','Joy','Love','Annoyed','Fear','Surprise']

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        load_model_and_tokenizer()
        video_url = trim_whitespace(request.form.get('video_url'))
        video_id = extract_youtube_video_id(video_url)

        if video_id:
            comments_df = get_comments(video_id,max_results=250)
            if not comments_df.empty:
                sentiment_list = get_sentiment(comments_df)
                print(sentiment_list)
                top_comment = comments_df.iloc[0]["Comment"]
                return render_template('index.html', comment=top_comment)
            else:
                return render_template('index.html', error="No comments found for this video.")
        else:
            return render_template('index.html', error="Invalid link, please try another link.")
    
    return render_template('index.html')

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

def get_sentiment(comments_df : pd.DataFrame) -> list:
    sentiment_list=[0,0,0,0,0,0]
    for index, row in comments_df.iterrows():
        translated_comment = detect_and_translate(row['Comment'])
        sentiment = return_sentiment(translated_comment)  
        sentiment_list[sentiment] += 1
    return sentiment_list

if __name__ == "__main__":
    app.run(debug=True)