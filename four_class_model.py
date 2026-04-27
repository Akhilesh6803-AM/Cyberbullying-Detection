import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

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


def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"(?:rt\s*)?@\w+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|t\.co/\S+|bit\.ly/\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if tokenizer is not None and stop_words:
        toks = tokenizer.tokenize(text)
        toks = [t for t in toks if t not in stop_words]
        text = " ".join(toks)

    return text


# load trained artifacts
tfidf: TfidfVectorizer = joblib.load("saved_models/tfidf_4class.pkl")
model = joblib.load("saved_models/model_4class.pkl")
label_names = joblib.load("saved_models/classes_4class.pkl")


def categorize_text(text: str):
    """
    Returns:
        label: one of
            - 'Normal'  // 0.3
            - 'Aggression'   //<.70
            - 'Bullying with low aggression'  //>.50
            - 'Bullying with high aggression'   //<.90
        confidence: float in [0, 1]
    """
    clean = basic_clean(text)
    X = tfidf.transform([clean])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # if no predict_proba, approximate from decision_function
        scores = model.decision_function(X)[0]
        exps = np.exp(scores - scores.max())
        probs = exps / exps.sum()

    idx = probs.argmax()
    return label_names[idx], float(probs[idx])
