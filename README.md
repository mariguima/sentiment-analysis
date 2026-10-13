# Sentiment Analysis Dashboard

A small web app for exploring the sentiment of product reviews. Enter a review in the Streamlit dashboard and the app predicts whether it is **positive** or **negative**, along with the model's confidence.

The classifier is a scikit-learn logistic regression model. It turns review text into TF-IDF features and is trained on a 20,000-review sample of the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Three-star reviews are excluded; reviews above three stars are labelled positive and those below three stars negative.

## Requirements

- Python 3.11 or newer
- Internet access for the first model-training run, so KaggleHub can download the dataset

## Set up and run

From the project directory, create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit scikit-learn kagglehub
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit scikit-learn kagglehub
```

Train the model and create the local model files:

```bash
python ml/train.py
```

This downloads the dataset through KaggleHub, trains and evaluates the classifier, then writes `saved_models/sentiment_model.pkl` and `saved_models/vectorizer.pkl`. The training step can take a few minutes depending on your computer and connection.

Start the dashboard:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually <http://localhost:8501>) to open in your browser. Type a review and select **Analyze**.

## Project layout

```text
app.py                         Streamlit user interface
ml/train.py                    Dataset download, preprocessing, training, evaluation
src/sentiment_dashboard/model.py Loads the saved model and predicts sentiment
saved_models/                  Generated model and vectorizer files
requirements.txt               Project's pinned Python dependencies
```

The saved model files are generated locally and are ignored by Git. If they are missing, run `python ml/train.py` before starting the dashboard.
