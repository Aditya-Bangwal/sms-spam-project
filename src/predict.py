import tensorflow as tf

import re
import string
import os



print(tf.__version__)
print(tf.keras)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models",  "spam_model.keras")

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
with open("models/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vectorizer.set_vocabulary(vocab)

# Load model
model=tf.keras.models.load_model(
     model_path,
    custom_objects={"custom_standardization": custom_standardization}
)

def predict_message(message):
    vectorized = vectorizer(tf.constant([message]))
    
    prediction = model.predict(vectorized, verbose=0)[0][0]
    print("Raw prediction:", prediction)

    return "SPAM" if prediction > 0.4 else "HAM"

# Test
if __name__ == "__main__":
    msg = input("Enter a message: ")
    result = predict_message(msg)
    print("Prediction:", result)
    