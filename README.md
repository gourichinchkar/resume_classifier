# Resume Classifier & Job Match Score

A machine learning project that classifies a resume into a job category and scores how well it matches a job description, with a list of keywords from the job description that are missing in the resume.

## Features
- Text cleaning with NLTK (stopwords removal, lemmatization)
- TF-IDF features (unigrams and bigrams, 3000 features)
- Compared three models: Logistic Regression, Linear SVM, Random Forest
- Job match score using cosine similarity between resume and job description
- Streamlit web app

## Results
Dataset: 169 resumes (small dataset, so results are indicative only).

| Model | Test accuracy | F1 (weighted) | 3-fold CV |
|---|---|---|---|
| Logistic Regression | 0.559 | 0.466 | 0.378 |
| Linear SVM | 0.794 | 0.763 | 0.756 |
| Random Forest | 0.706 | 0.636 | 0.600 |

Linear SVM performed best and was saved as the final model.

## Limitations
- Small dataset (169 resumes), so a larger dataset would improve accuracy.
- Match score is keyword-based (TF-IDF), not semantic. It measures word overlap, not meaning.

## Project structure
- `train.py`: cleaning, TF-IDF, model comparison, confusion matrix, saving the model
- `Step6_Match_scoring.py`: prediction and match score logic
- `app.py`: Streamlit app
- `model.pkl`, `tfidf.pkl`: saved model and vectorizer

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Tech stack
Python, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Streamlit
