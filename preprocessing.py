import re

REPLACEMENTS = {
    "govt": "government",
    "mahila": "girls",
    "sr sec": "senior secondary",
    "inst": "institute"
}

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)

    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text)

    return text.strip()