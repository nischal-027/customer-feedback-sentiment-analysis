# Customer Feedback Sentiment Analysis

An end-to-end Machine Learning and Natural Language Processing (NLP) project that analyzes customer reviews and classifies them into **Positive** or **Negative** sentiment using TF-IDF vectorization and Logistic Regression.

The project includes text preprocessing, model training, evaluation, model serialization, and an interactive web application built with Streamlit.

---

## Project Overview

Understanding customer opinions is important for businesses to improve their products and services. However, manually analyzing thousands of customer reviews can be time-consuming.

This project addresses that problem by developing a sentiment analysis system that automatically classifies customer reviews based on their textual content.

Users can enter a review through a web interface and receive:
- Predicted sentiment (Positive or Negative).
- Predicted probability for each sentiment class.
- A view of the preprocessed review.

## Key Features

- **Text Preprocessing:** Lowercasing, contraction expansion, URL and HTML removal, emoji removal, and whitespace normalization.
- **Tokenization:** Breaking reviews into individual words using NLTK.
- **Stopword Removal:** Removing common English stopwords while preserving important negation words.
- **Stemming:** Applying Porter Stemmer to normalize words.
- **Feature Extraction:** Converting text into numerical features using TF-IDF.
- **Model Comparison:** Evaluating Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
- **Sentiment Prediction:** Classifying customer reviews into Positive and Negative categories.
- **Interactive Interface:** Providing predictions through a Streamlit web application.

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| NLTK | Natural Language Processing |
| Scikit-learn | Feature extraction, model training and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Final sentiment classification model |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Matplotlib / Seaborn | Data visualization, if used in the notebook |

## Dataset

The project uses a customer review dataset containing approximately 60,000 records.

The dataset contains the following columns:

- `Id`: Unique review identifier.
- `Review`: Customer's written feedback.
- `Rating`: Customer rating on a scale of 1 to 5.

### Sentiment Labeling

The original ratings were converted into binary sentiment categories:

| Rating | Sentiment |
|---|---|
| 1 | Negative |
| 2 | Negative |
| 3 | Negative |
| 4 | Positive |
| 5 | Positive |

After removing missing values, the dataset contained **59,973 reviews**.

### Class Distribution

| Sentiment | Number of Reviews | Percentage |
|---|---:|---:|
| Positive | 38,005 | 63.37% |
| Negative | 21,968 | 36.63% |
| **Total** | **59,973** | **100%** |

**Note:** The sentiment labels are derived from star ratings rather than independently human-annotated sentiment labels. Therefore, they represent rating-based sentiment proxies.

## Machine Learning Workflow

The project follows a complete NLP and Machine Learning pipeline.

1. **Data Loading and Exploration**
   - Load the dataset.
   - Inspect the structure and data types.
   - Analyze rating distribution.
   - Handle missing values.

2. **Text Preprocessing**
   - Convert text to lowercase.
   - Expand contractions.
   - Remove URLs and HTML tags.
   - Remove emojis and special characters.
   - Normalize whitespace.
   - Tokenize the reviews.
   - Remove stopwords while preserving negation words.
   - Apply Porter Stemming.

3. **Feature Extraction**
   - Apply TF-IDF Vectorization.
   - Use unigrams and bigrams to capture individual words and two-word combinations.
   - Limit the vocabulary to 10,000 features.

4. **Model Training**
   - Split the dataset into training and testing sets.
   - Use stratified sampling to maintain class distribution.
   - Train and compare three classification algorithms.

5. **Model Evaluation**
   - Evaluate using accuracy, precision, recall, and F1-score.
   - Compare the performance of all three models.

6. **Model Deployment**
   - Save the trained Logistic Regression model and TF-IDF vectorizer using Joblib.
   - Load the saved artifacts in the Streamlit application.
   - Accept new reviews and generate sentiment predictions.

## Model Training and Evaluation

Three Machine Learning algorithms were evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

### TF-IDF Configuration

The following configuration was used for feature extraction:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

### Model Performance Comparison

The following results were obtained on the held-out test dataset:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 95% | 94% |
| Logistic Regression | **96%** | **95%** |
| Linear SVM | 95% | 95% |

### Selected Model: Logistic Regression

Logistic Regression was selected as the final model because it achieved the highest reported test accuracy and strong performance across both sentiment classes.

It also supports probability predictions through `predict_proba()`, allowing the application to display predicted class probabilities.

### Logistic Regression Classification Report

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.92 | 0.96 | 0.94 |
| Positive | 0.98 | 0.95 | 0.97 |
| **Accuracy** | | | **0.96** |
| Macro Average | 0.95 | 0.96 | 0.95 |
| Weighted Average | 0.96 | 0.96 | 0.96 |

The test set contained 11,995 reviews:
- Negative: 4,394
- Positive: 7,601

The Logistic Regression model was trained using class weighting to help account for the imbalance between the two sentiment classes.

Project Structure

Customer_review_analysis/
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── main.py
├── preprocessing.py
├── requirements.txt
├── README.md
└── .gitignore

File Descriptions

File

Description

main.py

Streamlit application and sentiment prediction

preprocessing.py

Text cleaning, tokenization, stopword removal and stemming

sentiment_model.pkl

Trained Logistic Regression model

tfidf_vectorizer.pkl

Fitted TF-IDF vectorizer

requirements.txt

Python project dependencies

.gitignore

Files and folders excluded from Git

Installation and Setup

Follow these steps to run the project locally.

1. Clone the Repository

git clone https://github.com/nischal-027/customer-feedback-sentiment-analysis.git

2. Navigate to the Project Directory

cd customer-feedback-sentiment-analysis

3. Create a Virtual Environment

python -m venv .venv

4. Activate the Virtual Environment

For Windows PowerShell:

.venv\Scripts\Activate.ps1

5. Install Dependencies

pip install -r requirements.txt

6. Run the Streamlit Application

streamlit run main.py

The application will open in your browser, usually at:

http://localhost:8501

## Application Preview

### Positive Sentiment Prediction

![Positive Sentiment](screenshots/positive.png)

### Negative Sentiment Prediction

![Negative Sentiment](screenshots/negative.png)


Future Improvements

Some potential improvements for future development include:

Experimenting with Lemmatization and comparing it with Stemming.

Exploring Word2Vec and other word embedding techniques.

Experimenting with deep learning models such as LSTM and BiLSTM.

Exploring transformer-based sentiment classification models.

Adding sentiment distribution dashboards and customer feedback analytics.

Deploying the application online for public access.

Improving prediction reliability through additional validation and error analysis.

Learning Outcomes

Through this project, I gained practical experience in:

Natural Language Processing and text preprocessing.

Converting unstructured text into numerical features.

Training and comparing Machine Learning classification models.

Evaluating models using multiple classification metrics.

Saving and deploying trained Machine Learning models.

Building interactive applications using Streamlit.

Managing project dependencies and version control using Git and GitHub.

Author

Nischal Acharya

GitHub: @nischal-027

If you find this project useful, feel free to explore the repository and provide feedback!
