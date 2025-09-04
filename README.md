# Movie Sentiment Classification

A robust pipeline for movie review sentiment analysis using modern NLP and machine learning techniques.  
This project demonstrates preprocessing, embedding, sentiment labeling, and SVM classification for movie reviews.

---

## Features

- **Text Preprocessing:** HTML cleaning, emoji removal, punctuation and stopword filtering, lemmatization.
- **Sentence Embeddings:** Uses [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for high-quality text embeddings.
- **Sentiment Labeling:** Assigns positive, neutral, or negative sentiment using embedding similarity.
- **SVM Classification:** Trains a Support Vector Machine classifier for sentiment prediction.
- **Evaluation & Visualization:** Model metrics, confusion matrix, confidence analysis.
- **Exception Handling:** Custom error handler for robust pipeline execution.

---

## Project Structure

```
rapidAI/
│
├── movie.csv                       # Input data (movie reviews)
├── notes.ipynb                     # Main Jupyter notebook (pipeline code)
├── requirements.txt                # Python dependencies
├── svm_test_predictions.csv        # SVM test predictions (output)
├── svm_sentiment_classifier.pkl    # Trained SVM model (output)
├── text_embeddings.npy             # Saved sentence embeddings (output)
├── svm_model_metrics.csv           # Model metrics (output)
└── movie_reviews_with_sentiment.csv# Sentiment-labeled data (output)
```

---

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <your-repo-url>
   cd rapidAI
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   .\venv\Scripts\activate      # Windows
   # Or
   source venv/bin/activate    # macOS/Linux
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download Spacy English model:**
   ```
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK resources (run in notebook or Python shell):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   # Optional:
   # nltk.download('wordnet')
   # nltk.download('omw-1.4')
   ```

---

## Usage

- Open `notes.ipynb` in Jupyter or VS Code.
- Run all cells to preprocess data, generate embeddings, label sentiment, train SVM, and evaluate.
- Outputs (CSV, model, metrics) are saved in the project folder.

---

## Custom Exception Handling

See [`src/exception_handler.py`](src/exception_handler.py) for a reusable decorator and handler for robust error logging.

---

## Example Results

- Sentiment distribution and sample predictions are printed in the notebook.
- Confusion matrix and confidence analysis plots are saved as PNG files.

---

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## License

This project is for educational and research purposes.

---

## Author