import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score


INPUT_FILE = "evaluation/amazon_labeling_300_completed.xlsx"


print("Loading labeled data...")

df = pd.read_excel(INPUT_FILE)

# Remove rows with missing labels
df = df.dropna(subset=["intent"])

X = df["customer_text"].astype(str)
y = df["intent"].astype(str)

print(f"Total labeled examples: {len(df)}")


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")


# TF-IDF + Logistic Regression
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining model...")
model.fit(X_train, y_train)

print("Training complete.")


# Predictions
y_pred = model.predict(X_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")


print("\n==============================")
print("TF-IDF + Logistic Regression")
print("==============================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))