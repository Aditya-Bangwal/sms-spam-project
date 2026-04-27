from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import re
import string
import os

from pydantic import BaseModel







# Load model






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

from keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )


vectorizer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)

# load vocab
vocab_path = os.path.join(BASE_DIR, "models", "vocab.txt")

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vectorizer.set_vocabulary(vocab)

print("✅ VECTORIZER LOADED")




print("🚀 APP STARTING...")

print("📂 BASE_DIR:", BASE_DIR)
print("📂 MODEL PATH:", model_path)
print("📂 FILE EXISTS:", os.path.exists(model_path))

try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"custom_standardization": custom_standardization}
    )
    print("✅ MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("❌ MODEL LOAD FAILED:", e)
    raise e


@app.get("/")
def home():
    return {"message": "Spam Classifier API Running"}

class MessageRequest(BaseModel):
    message: str
@app.post("/predict")
def predict(data: MessageRequest):
    message = data.message

    vectorized = vectorizer([message])
    prediction = model.predict(vectorized, verbose=0)[0][0]

    result = "SPAM" if prediction > 0.4 else "HAM"

    return {
        "message": message,
        "prediction": result,
        "confidence": float(prediction)
    }