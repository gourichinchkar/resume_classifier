import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_resumes.csv")

X = df["Cleaned"]
y = df["Category"]

# Dataset chhota aahe (169 rows), mhanun check karto ki stratify chalel ka
min_count = y.value_counts().min()
print("Sarvat kami resumes ek category madhe:", min_count)

stratify = y if min_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("Train shape:", X_train_vec.shape)
print("Test shape:", X_test_vec.shape)
print("Sample features:", tfidf.get_feature_names_out()[:15])