# 📰 Fake News Detector

A Natural Language Processing (NLP) project that uses **TF-IDF vectorization** and **Logistic Regression** to classify news articles as **FAKE** or **REAL** with over **98% accuracy**.

---

## 📊 Dataset

- **Source**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Total Articles**: ~24,000
- **Classes**:
  - `FAKE`: Articles flagged as misinformation
  - `REAL`: Verified articles from legitimate sources

---

## 🚀 Project Highlights

- 🧹 **Data Cleaning**:
  - Lowercasing
  - Removing punctuation, URLs, and numbers
  - Stopword removal
  - Lemmatization (with POS tagging)

- 🔢 **Feature Extraction**:
  - `TfidfVectorizer` with `max_features=5000`
  - Transforms text into numerical vectors

- 🤖 **Model**:
  - `LogisticRegression` (fast and interpretable)
  - Trained on 80% of the data, tested on 20%

- 📈 **Performance**:
  - **Accuracy**: `98.63%`
  - **F1-Score**: `0.99` (macro average)
  - **Confusion Matrix**:
  
    |               | Predicted FAKE | Predicted REAL |
    |---------------|----------------|----------------|
    | **Actual FAKE** |     4618       |       78        |
    | **Actual REAL** |      45        |      4239       |

---

## 💾 Model Files

Model and vectorizer are saved as:

- `tfidf_vectorizer.joblib`
- `fake_news_detection_model.joblib`

These are stored in **Google Drive** (not committed to the repo).

---

## 📂 How to Use

> Run in Google Colab or locally with Jupyter:

1. Install required libraries:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn

