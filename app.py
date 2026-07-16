import os
import re
import sqlite3
import emoji
import contractions
import numpy as np
from four_class_model import categorize_text
from datetime import datetime
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import joblib

# Lazy imports for PyTorch and Transformers to allow deployment without them
torch = None
AutoTokenizer = None
AutoModelForSequenceClassification = None

def _lazy_import_bert_deps():
    global torch, AutoTokenizer, AutoModelForSequenceClassification
    if torch is None:
        try:
            import torch as _torch
            from transformers import AutoTokenizer as _AutoTokenizer, AutoModelForSequenceClassification as _AutoModelForSequenceClassification
            torch = _torch
            AutoTokenizer = _AutoTokenizer
            AutoModelForSequenceClassification = _AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "PyTorch ('torch') and Hugging Face Transformers ('transformers') are required to run the BERT model. "
                "Please install them or change your model configuration to SVM or NaiveBayes."
            )

# =========================
# Basic Flask setup
# =========================
app = Flask(__name__)
app.secret_key = "CHANGE_THIS_SECRET_KEY"  # 🔒 change in production

DB_NAME = "cyberbullying.db"
SAVED_MODELS_DIR = "saved_models"

# =========================
# NLTK setup (for cleaning)
# =========================
nltk.download("stopwords", quiet=True)
tokenizer_nltk = TweetTokenizer()
stop_words = set(stopwords.words("english"))
stop_words.update(["rt", "u"])

# =========================
# Text cleaning (same logic style as train.py)
# =========================
def expand_hashtag(text):
    hashtags = re.compile(r"#(\w+)")

    def replace_hashtag(match):
        return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", match.group(1))

    return hashtags.sub(replace_hashtag, text)


def cleaning_data(text):
    text = str(text)

    text = contractions.fix(text)
    text = expand_hashtag(text)
    text = text.lower()

    # remove mentions & RT
    text = re.sub(r"(?:rt\s*)?@\w+", "", text)
    # remove urls
    text = re.sub(r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+", "", text)

    # emojis to text
    text = emoji.demojize(text)

    # keep letters, spaces, colon
    text = re.sub(r"[^a-zA-Z\s:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_remove_stopwords(text):
    tokens = tokenizer_nltk.tokenize(text)
    clean_tokens = [tok for tok in tokens if tok not in stop_words]
    return " ".join(clean_tokens)


# =========================
# Model loader
# =========================
class CyberbullyingModel:
    def __init__(self):
        meta_path = os.path.join(SAVED_MODELS_DIR, "model_meta.txt")
        if not os.path.exists(meta_path):
            raise RuntimeError(
                "model_meta.txt not found. Run train.py to generate models first."
            )

        with open(meta_path, "r") as f:
            line = f.read().strip()

        # expects: best_model=NaiveBayes / SVM / BERT
        if "=" in line:
            self.best_model = line.split("=", 1)[1].strip()
        else:
            self.best_model = line.strip()

        print(f"[INFO] Best model from meta: {self.best_model}")

        self.le = joblib.load(os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl"))

        if self.best_model == "NaiveBayes":
            self._load_nb()
        elif self.best_model == "SVM":
            self._load_svm()
        elif self.best_model == "BERT":
            self._load_bert()
        else:
            print("[WARN] Unknown best_model, defaulting to Naive Bayes if available.")
            self._load_nb()

    # ---------- Naive Bayes ----------
    def _load_nb(self):
        self.model = joblib.load(os.path.join(SAVED_MODELS_DIR, "nb_model.pkl"))
        self.tfidf = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
        self.mode = "CLASSIC"
        print("[INFO] Loaded Naive Bayes model.")

    # ---------- SVM ----------
    def _load_svm(self):
        self.model = joblib.load(os.path.join(SAVED_MODELS_DIR, "svm_model.pkl"))
        self.tfidf = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
        self.mode = "CLASSIC"
        print("[INFO] Loaded SVM model.")

    # ---------- BERT ----------
    def _load_bert(self):
        _lazy_import_bert_deps()
        save_dir = os.path.join(SAVED_MODELS_DIR, "bert_cyberbullying")
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        self.model.eval()
        self.mode = "BERT"
        print("[INFO] Loaded DistilBERT model from saved_models.")

    # ---------- Predict ----------
    def predict(self, text: str):
        clean = cleaning_data(text)

        # If classic model (NB/SVM using TF-IDF)
        if self.mode == "CLASSIC":
            tok = tokenize_and_remove_stopwords(clean)
            X = self.tfidf.transform([tok])

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
            else:
                # For SVM without probability=True, approximate via decision_function
                scores = self.model.decision_function(X)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()

            idx = int(np.argmax(probs))
            label = self.le.inverse_transform([idx])[0]
            confidence = float(probs[idx])
            return label, confidence

        # If BERT model
        elif self.mode == "BERT":
            _lazy_import_bert_deps()
            encoding = self.tokenizer(
                clean,
                add_special_tokens=True,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            idx = int(np.argmax(probs))
            label = self.le.inverse_transform([idx])[0]
            confidence = float(probs[idx])
            return label, confidence

        else:
            raise RuntimeError("Model mode not configured.")


# create global model instance
# model_wrapper = CyberbullyingModel()  # Commented out to save memory (unused; routes use categorize_text instead)

# =========================
# SQLite helpers
# =========================
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Predictions history table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )

    conn.commit()
    conn.close()


# Initialize database and directories on import (needed for WSGI servers like Gunicorn)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
init_db()


# =========================
# Auth utilities
# =========================
def current_user():
    if "user_id" in session:
        return {
            "id": session["user_id"],
            "username": session.get("username"),
        }
    return None


def login_required(view_func):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    wrapper.__name__ = view_func.__name__
    return wrapper


# =========================
# Routes
# =========================
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route('/home')
@login_required
def home():
    return render_template('home.html', user=current_user())


@app.route("/detector")
@login_required
def detector():
    return render_template("index.html", user=current_user())


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Please enter a message to analyze.", "danger")
        return redirect(url_for("detector"))

    label, confidence = categorize_text(text)

    # save to SQLite as before
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (user_id, input_text, prediction_label, confidence, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (session["user_id"], text, label, confidence),
        )
        conn.commit()
    finally:
        conn.close()

    return render_template(
        "index.html",
        user=current_user(),
        input_text=text,
        prediction=label,
        confidence=round(confidence * 100, 2),
    )


@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT input_text, prediction_label, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
            """,
            (session["user_id"],),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return render_template("history.html", user=current_user(), rows=rows)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash),
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Choose another.", "danger")
            return redirect(url_for("register"))
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            )
            user = cur.fetchone()
        finally:
            conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# =========================
# Main
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
