
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("Resume.csv")

df = df[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str).str.lower()
df["Resume_str"] = df["Resume_str"].apply(
    lambda text: re.sub(r"\s+", " ", text).strip()
)
df = df[df["Resume_str"] != ""]

print("Dataset shape:", df.shape)
print("\nNumber of categories:", df["Category"].nunique())
print("\nCategory distribution:")
print(df["Category"].value_counts())

X = df["Resume_str"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation="vertical",
    cmap="Blues",
    colorbar=False
)
plt.title("Resume Category Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

joblib.dump(model, "resume_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as resume_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
print("Confusion matrix saved as confusion_matrix.png")
