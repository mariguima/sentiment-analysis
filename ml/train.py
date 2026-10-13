import os
import joblib
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_data():
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    csv_path = os.path.join(path, "Reviews.csv")
    df = pd.read_csv(csv_path)
    return df


def preprocess(df):
    # Drop neutral (3-star) reviews to keep it a clean binary problem
    df = df[df["Score"] != 3].copy()
    df["sentiment"] = df["Score"].apply(lambda x: "positive" if x > 3 else "negative")

    # Keep it manageable for a first pass — bump this up later if you want
    df = df.sample(n=20000, random_state=42)

    return df["Text"], df["sentiment"]


def train():
    print("Downloading + loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("\nEvaluation:")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(SAVED_MODELS_DIR, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl"))
    print(f"\nSaved model + vectorizer to {SAVED_MODELS_DIR}")


if __name__ == "__main__":
    train()