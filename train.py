import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional: use NLTK stopwords if you like; keeping it simple here.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
    nltk.download("stopwords", quiet=True)
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(["rt", "u"])
except Exception:
    tokenizer = None
    stop_words = set()


# ---- Severity lexicon (you can adjust) ----
HIGH_AGGR_TERMS = {
    "fuck", "fucking", "bitch", "bastard", "whore","slut",
    "retard", "dumbass", "moron", "kys", "kill yourself",
    "die", "ugly", "stupid", "kill", "murder", "threat"
}


def basic_clean(text: str) -> str:
    text = str(text).lower()

    # remove mentions & urls & hashtags
    text = re.sub(r"(?:rt\s*)?@\w+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|t\.co/\S+|bit\.ly/\S+", " ", text)
    text = re.sub(r"#", " ", text)

    # keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # optional stopword removal if NLTK available
    if tokenizer is not None and stop_words:
        tokens = tokenizer.tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        text = " ".join(tokens)

    return text


def has_high_aggression(text: str) -> bool:
    t = str(text).lower()
    # To prevent "kill" from matching "skill", pad with spaces or use regex word boundaries
    # A simple trick: pad the text with spaces
    padded_t = f" {t} "
    
    for term in HIGH_AGGR_TERMS:
        # pad term with spaces to ensure full word match, or just use regex boundary
        if re.search(r'\b' + re.escape(term) + r'\b', t):
            return True
    return False


def map_to_4class(row) -> str:
    t = row["tweet_text"]
    ctype = row["cyberbullying_type"]

    # Explicit override for highly aggressive terms regardless of dataset category
    if has_high_aggression(t):
        return "Bullying with high aggression"

    if ctype == "not_cyberbullying":
        return "Normal"

    # Generic cyberbullying
    if ctype == "other_cyberbullying":
        return "Aggression"

    # Targeted cyberbullying: age / gender / religion / ethnicity
    if ctype in ["age", "gender", "religion", "ethnicity"]:
        return "Bullying with low aggression"

    # Fallback (shouldn't happen)
    return "Normal"


def main():
    df = pd.read_csv("cyberbullying_tweets.csv")

    if "tweet_text" not in df.columns or "cyberbullying_type" not in df.columns:
        raise ValueError("CSV must contain 'tweet_text' and 'cyberbullying_type' columns.")

    # create new 4-class label
    df["label_4class"] = df.apply(map_to_4class, axis=1)

    # clean text
    df["clean_text"] = df["tweet_text"].astype(str).apply(basic_clean)

    # remove empty
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    LABELS = [
        "Normal",
        "Aggression",
        "Bullying with low aggression",
        "Bullying with high aggression",
    ]

    label_to_id = {lbl: i for i, lbl in enumerate(LABELS)}
    df["y"] = df["label_4class"].map(label_to_id)

    X = df["clean_text"].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(
        max_iter=400,
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)

    # quick eval in console
    y_pred = clf.predict(X_test_vec)
    print(classification_report(
        y_test,
        y_pred,
        target_names=LABELS,
        digits=4
    ))

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(tfidf, "saved_models/tfidf_4class.pkl")
    joblib.dump(clf, "saved_models/model_4class.pkl")
    joblib.dump(LABELS, "saved_models/classes_4class.pkl")
    print("Saved: saved_models/tfidf_4class.pkl, model_4class.pkl, classes_4class.pkl")


if __name__ == "__main__":
    main()
