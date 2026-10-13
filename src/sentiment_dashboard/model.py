import os
import joblib

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models")

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = os.path.join(SAVED_MODELS_DIR, "sentiment_model.pkl")
        vectorizer_path = os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "No trained model found. Run `python ml/train.py` first."
            )

        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vectorizer_path)


def predict(text: str) -> dict:
    _load()

    text_vec = _vectorizer.transform([text])
    label = _model.predict(text_vec)[0]
    proba = _model.predict_proba(text_vec)[0]

    class_index = list(_model.classes_).index(label)
    confidence = proba[class_index]

    return {"label": label, "confidence": round(float(confidence), 3)}