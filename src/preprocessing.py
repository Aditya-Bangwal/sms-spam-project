import re
import string
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text