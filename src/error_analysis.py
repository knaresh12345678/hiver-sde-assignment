import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


INPUT_FILE = "evaluation/amazon_labeling_300_completed.xlsx"

print("Loading labeled data...")

df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=["customer_text", "intent"])

X = df["customer_text"].astype(str)
y = df["intent"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n==============================")
print("INCORRECT PREDICTIONS")
print("==============================")

errors = []

for text, actual, predicted in zip(X_test, y_test, y_pred):

    if actual != predicted:
        errors.append({
            "actual": actual,
            "predicted": predicted,
            "message": text
        })

print(f"\nTotal test examples: {len(y_test)}")
print(f"Incorrect predictions: {len(errors)}")

for i, error in enumerate(errors, start=1):

    print(f"\n{i}.")
    print(f"ACTUAL    : {error['actual']}")
    print(f"PREDICTED : {error['predicted']}")
    print(f"MESSAGE   : {error['message']}")
    print("-" * 70)