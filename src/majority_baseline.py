import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT_FILE = "evaluation/amazon_labeling_300_completed.xlsx"

print("Loading labeled data...")

df = pd.read_excel(INPUT_FILE)
df = df.dropna(subset=["intent"])

X = df["customer_text"].astype(str)
y = df["intent"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Find the most common intent in training data
majority_intent = y_train.value_counts().idxmax()

# Predict the same intent for every test example
y_pred = [majority_intent] * len(y_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n==============================")
print("Majority Class Baseline")
print("==============================")

print(f"Majority intent: {majority_intent}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)