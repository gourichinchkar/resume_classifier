import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("cleaned_resumes.csv")
X, y = df["Cleaned"], df["Category"]

stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# Best model nivadha (test accuracy var)
best_name, best_model, best_acc = None, None, -1
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print("Best model:", best_name, "| accuracy:", round(best_acc, 3))

pred = best_model.predict(X_test_vec)
print(classification_report(y_test, pred, zero_division=0))

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, ax=ax, xticks_rotation=90, colorbar=False
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Model ani TF-IDF save kara (Streamlit app sathi lagtil)
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("Saved: model.pkl, tfidf.pkl, confusion_matrix.png")