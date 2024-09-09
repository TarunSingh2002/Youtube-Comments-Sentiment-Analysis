# Youtube Comments Sentiment Analysis

**[Check out the live project here!](https://tarun-singh-youtube-video-comments-sentiment-analysis.hf.space)**

Welcome to the YouTube Comment Emotion Analyzer! This project leverages deep learning transformers to classify YouTube comments into six distinct emotions.

<table border="2" style="width:100%; border-collapse: collapse;">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b5d9b774-0695-4cf9-912c-c93e052701ca" alt="Project Image 1" style="width:100%;"></td>
  </tr>
</table>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

## Project Overview

The YouTube Comment Emotion Analyzer is a machine learning-based application designed to classify emotions in YouTube comments. It utilizes a pre-trained transformer model (MobileBERT) fine-tuned on the DAIR AI Emotion dataset, which is then optimized and deployed in TensorFlow Lite format. The model processes comments, translates them to English (if necessary), and classifies them into one of six emotions: Sadness, Joy, Love, Annoyed, Fear, and Surprise. The application is hosted via a web interface and deployed on Hugging Face Spaces for easy access.

## Features

- **MobileBERT Transformer Model:** A compact, pre-trained transformer model fine-tuned on the DAIR AI Emotion dataset to classify emotions with high accuracy.
- **TensorFlow Lite with Float16 Quantization:** The model is converted to TensorFlow Lite format with Float16 Quantization to ensure it is optimized for performance in real-time applications.
- **YouTube API Integration:** Fetches YouTube comments in real-time, processing them for emotional analysis.
- **Translation Module:** Automatically translates non-English comments into English before analysis to ensure accurate emotion detection.
- **Emotion Classification:** Classifies comments into six emotions—Sadness, Joy, Love, Annoyed, Fear, and Surprise—and displays the count for each.
- **Flask Web Interface:** A user-friendly web interface built using Flask, HTML, CSS, and JavaScript allows users to input a YouTube video URL and retrieve classified comment results.
- **Dockerized:** The entire application is containerized using Docker for consistent deployment and scalability.
- **Deployed on Hugging Face Spaces:** Easily accessible as a web app hosted on Hugging Face Spaces for seamless user experience.

## Project Structure

```markdown
├── .dvc                         <- DVC configuration file for data version control.
├── notebook                     <- Source code used in this project.
│   ├── converting-model.ipynb   <- 
│   ├── model-training.ipynb     <- 
│   └── accuracy.ipynb           <- 
├── templates                    <-
│   └── index.html               <- HTML template for the web app.
├── .dvcignore                   <- Specifies which files to ignore in the version control.
├── .gitattributes               <- 
├── .gitignore                   <- Specifies which files to ignore in the version control.
├── app.py                       <- Main Flask application file.
├── Dockerfile                   <- Dockerfile for containerizing the app.
├── evaluation_results1          <- store accuracy and loss of tarnsformer model 
├── evaluation_results2          <- store accuracy of tflite model
├── LICENSE                      <- DVC ignore.
├── model.dvc                    <- 
├── README.md                    <- The top-level README for developers using this project.
├── requirements.txt             <- The requirements file for reproducing the environment.

## License

```markdown
MIT License

Copyright (c) 2024 Tarun Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

This project would not have been possible without the works of **J.K. Rowling** and her creation of the Harry Potter series. The text from the seven books serves as the foundation for the vector database used in this application. Special thanks to the entire **Harry Potter fandom** for keeping the magic alive and inspiring projects like this one.