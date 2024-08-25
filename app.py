import os
import re
import awsgi
import pandas as pd
from dotenv import load_dotenv
from googletrans import Translator
from googleapiclient.discovery import build
from flask import Flask, render_template, request, render_template_string

app = Flask(__name__)

load_dotenv()
api_keys = os.getenv("YOUTUBE_API_KEYS").split(',')

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
        video_url = trim_whitespace(request.form.get('video_url'))
        video_id = extract_youtube_video_id(video_url)

        if video_id:
            comments_df = get_comments(video_id)
            if not comments_df.empty:
                top_comment = comments_df.iloc[0]["Comment"]
                return render_template('index.html', comment=top_comment)
            else:
                return render_template('index.html', error="No comments found for this video.")
        else:
            return render_template('index.html', error="Invalid link, please try another link.")
    
    return render_template('index.html')

def detect_and_translate(text):
    translator = Translator()
    
    detection = translator.detect(text)
    
    if detection.lang != 'en':
        translation = translator.translate(text, dest='en')
        return translation.text
    else:
        return text

if __name__ == "__main__":
    app.run(debug=True)