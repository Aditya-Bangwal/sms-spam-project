# sms-spam-project
Overview
Spam messages are a common problem that waste time and can lead to fraud. This project builds an end-to-end SMS Spam Detection System using Machine Learning, with a deployed web interface where users can classify messages in real time.
The system takes an SMS as input and predicts whether it is Spam or Ham (Not Spam).

Problem Statement
With the increasing volume of SMS communication, detecting spam manually is inefficient. This project aims to automate spam detection using NLP and Machine Learning techniques.

Approach
1. Data Preprocessing
Lowercasing text
Removing punctuation and special characters
Tokenization
Removing stopwords
Text vectorization using:
TF-IDF / TextVectorization layer (update based on your implementation)

2. Model
Model Used: (Update this — e.g., LSTM / Logistic Regression / Naive Bayes)
Framework: TensorFlow / Scikit-learn
Trained on labeled SMS dataset

Live Demo
Deployed Link: [ https://sms-spam-project-1.onrender.com ]

Tech Stack
Frontend: React (Vite)
Backend:FastAPI
Machine Learning:TensorFlow / Scikit-learn ,NumPy, Pandas
Deployment: Render (Backend + Frontend)

Dataset
Dataset Used: SMS Spam Collection Dataset
Contains labeled SMS messages as spam or ham

Features
Real-time spam detection
Clean and simple UI
Fast API response
End-to-end ML pipeline


Future Improvements
Add probability score for predictions
Improve model accuracy with advanced architectures
Add explainability (why a message is spam)
Store prediction history

Author-Aditya bangwal

Acknowledgements:Open datasets and ML community resources
