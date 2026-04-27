import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from sklearn.utils import class_weight
import numpy as np
import os

os.environ["PYTHONUTF8"] = "1"
import re
import shutil

print(tf.__version__)
print(tf.keras)

import string


@register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )

def remove_bad_chars(text):
    return re.sub(r"[^\x00-\x7F]+", " ", str(text))




train_df = pd.read_csv("data/train-data.tsv", sep="\t", header=None, encoding='utf-8')
test_df = pd.read_csv("data/valid-data.tsv", sep="\t", header=None, encoding='utf-8')


train_df.columns = ['label', 'message']
test_df.columns = ['label', 'message']


train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

print(train_df.head())
print(train_df['label'].value_counts())


train_df['message'] = train_df['message'].apply(remove_bad_chars)
test_df['message'] = test_df['message'].apply(remove_bad_chars)
X_train = train_df['message'].astype(str)
y_train = train_df['label']

X_test = test_df['message'].astype(str)
y_test = test_df['label']


vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,   
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)

class_weights = dict(enumerate(class_weights))


vectorize_layer.adapt(X_train.values)

vocab = vectorize_layer.get_vocabulary()

with open("models/vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")


model = tf.keras.Sequential([
   
    tf.keras.layers.Embedding(10000, 128),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


X_train_vec = vectorize_layer(train_df['message'].values)
X_test_vec = vectorize_layer(test_df['message'].values)

model.fit(
    X_train_vec,
    train_df['label'].values,
    epochs=12,
    validation_data=(X_test_vec, test_df['label'].values),
    class_weight=class_weights
)


# model_dir = "models/spam_model.keras"
# if os.path.exists(model_dir):
#     shutil.rmtree(model_dir, ignore_errors=True)


# model.save(model_dir)

model.save("models/spam_model.h5")

