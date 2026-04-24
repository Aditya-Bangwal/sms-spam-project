from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import re
import string
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "spam_model.keras")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAME function (required again)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )

# Load model
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"custom_standardization": custom_standardization}
)


@app.get("/")
def home():
    return {"message": "Spam Classifier API Running"}

@app.post("/predict")
def predict(message: str):
    prediction = model.predict(tf.constant([message]))[0][0]
    result = "SPAM" if prediction > 0.4 else "HAM"
    
    return {
        "message": message,
        "prediction": result,
        "confidence": float(prediction)
    }