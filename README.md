# Youtube Comments Sentiment Analysis

**[Check out the live project here!](https://tarun-singh-youtube-video-comments-sentiment-analysis.hf.space)**

Welcome to the YouTube Comment Sentiment Analyzer! This project leverages deep learning transformers to classify YouTube comments into six distinct emotions.

<p align="center" float="left">
<table>
  <tr>
    <td>Application</td> 
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/cb12247c-f07a-4650-a281-f2d13a71843a" style="width:100%;"></video></td>
  </tr>
 </table>
 </p>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

## Project Overview

The YouTube Comment Emotion Analyzer is a deep learning application that uses the YouTube API to fetch real-time comments from any video and classify them into six emotions: Sadness, Joy, Love, Annoyed, Fear, and Surprise. It leverages a MobileBERT transformer model fine-tuned on the DAIR AI Emotion dataset, further optimized using Float16 Quantization for enhanced performance. The model is deployed in TensorFlow Lite format to ensure efficient classification. The application is built with Flask, containerized with Docker, and deployed on Hugging Face Spaces.

## Features

- **MobileBERT Transformer Model:** A compact, pre-trained transformer model fine-tuned on the DAIR AI Emotion dataset.
- **TensorFlow Lite with Float16 Quantization:** The model is converted to TensorFlow Lite format with Float16 Quantization to ensure it is optimized for performance in real-time applications.
- **Flask Web Interface:** A clean, easy-to-use web app built with Flask, using HTML, CSS, and JavaScript.
- **Dockerized:** The entire application is containerized using Docker for consistent deployment and scalability.
- **Deployed on Hugging Face Spaces:** Easily accessible as a web app hosted on Hugging Face Spaces for seamless user experience.

## Project Structure

```markdown
├── .dvc                         <- DVC configuration file for managing dataset version control.
├── notebook                     <- Jupyter Notebooks used for training and converting the model.
│   ├── converting-model.ipynb   <- Notebook for converting the model to TensorFlow Lite format.
│   ├── model-training.ipynb     <- Notebook used for training the transformer model on the emotion dataset.
│   └── accuracy.ipynb           <- Notebook for analyzing the accuracy of both the transformer and TFLite models.
├── templates                    <- HTML template for the Flask web application.
│   └── index.html               <- Main interface template for the web app.
├── .dvcignore                   <- File specifying files to ignore for dataset version control.
├── .gitattributes               <- Git configuration for handling file attributes and version control.
├── .gitignore                   <- File specifying which files and directories to ignore for version control.
├── app.py                       <- The main Flask application file that handles request routing and logic.
├── Dockerfile                   <- Docker configuration file for containerizing the app.
├── evaluation_results1          <- Contains evaluation results (accuracy, loss) of the transformer model.
├── evaluation_results2          <- Contains evaluation results for the TensorFlow Lite model.
├── LICENSE                      <- License for the project.
├── model.dvc                    <- DVC file for managing model version control.
├── README.md                    <- The main README file providing an overview and instructions for developers.
├── requirements.txt             <- List of dependencies needed to run the application.
```

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

This project owes its success to the invaluable resources provided by Hugging Face Spaces for hosting the application, and Hugging Face Transformers for the pre-trained MobileBERT model. Special thanks to the creators of the DAIR AI Emotion Dataset for supplying the essential data used for emotion classification, and to the YouTube API for enabling seamless retrieval of real-time video comments, which made the emotional analysis possible.
