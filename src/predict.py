import tensorflow as tf
import numpy as np
import re
import string
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "spam_model.keras")

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

def predict_message(message):
    prediction = model.predict(tf.constant([message]))[0][0]
    print("Raw prediction:", prediction)

    if prediction > 0.4:
        return "SPAM"
    else:
        return "HAM"

# Test
if __name__ == "__main__":
    msg = input("Enter a message: ")
    result = predict_message(msg)
    print("Prediction:", result)
    