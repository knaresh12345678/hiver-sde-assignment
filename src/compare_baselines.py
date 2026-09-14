import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


GOLDEN_FILE = "evaluation/golden_set_200_verified.xlsx"
LABELED_FILE = "evaluation/amazon_labeling_300_completed.xlsx"
OUTPUT_FILE = "results/baseline_comparison.csv"


def main():

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print("Loading Golden Evaluation Set...")

    golden = pd.read_excel(
        GOLDEN_FILE,
        sheet_name="Golden Set"
    )

    golden = golden.dropna(
        subset=["customer_text", "intent"]
    ).reset_index(drop=True)

    print(f"Golden examples: {len(golden)}")

    print()
    print("Loading labeled training data...")

    train = pd.read_excel(
        LABELED_FILE
    )

    train = train.dropna(
        subset=["customer_text", "intent"]
    ).reset_index(drop=True)

    print(f"Training examples: {len(train)}")

    X_train = train["customer_text"].astype(str)
    y_train = train["intent"].astype(str)

    X_test = golden["customer_text"].astype(str)
    y_test = golden["intent"].astype(str)

    # ---------------------------------------------------------
    # BASELINE 1 — MAJORITY CLASS
    # ---------------------------------------------------------

    majority_intent = y_train.value_counts().idxmax()

    majority_predictions = [
        majority_intent
        for _ in range(len(y_test))
    ]

    majority_accuracy = accuracy_score(
        y_test,
        majority_predictions
    )

    majority_macro_f1 = f1_score(
        y_test,
        majority_predictions,
        average="macro",
        zero_division=0
    )

    # ---------------------------------------------------------
    # BASELINE 2 — TF-IDF + LOGISTIC REGRESSION
    # ---------------------------------------------------------

    print()
    print("Training TF-IDF + Logistic Regression...")

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

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    tfidf_accuracy = accuracy_score(
        y_test,
        predictions
    )

    tfidf_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    # ---------------------------------------------------------
    # IMPROVEMENT
    # ---------------------------------------------------------

    accuracy_improvement = (
        tfidf_accuracy
        - majority_accuracy
    )

    macro_f1_improvement = (
        tfidf_macro_f1
        - majority_macro_f1
    )

    # ---------------------------------------------------------
    # CREATE COMPARISON TABLE
    # ---------------------------------------------------------

    comparison = pd.DataFrame([
        {
            "model": "Majority Class Baseline",
            "accuracy": majority_accuracy,
            "macro_f1": majority_macro_f1
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": tfidf_accuracy,
            "macro_f1": tfidf_macro_f1
        }
    ])

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Majority baseline intent: "
        f"{majority_intent}"
    )

    print()

    print(
        f"Majority baseline accuracy: "
        f"{majority_accuracy:.4f}"
    )

    print(
        f"Majority baseline Macro F1: "
        f"{majority_macro_f1:.4f}"
    )

    print()

    print(
        f"TF-IDF + Logistic Regression accuracy: "
        f"{tfidf_accuracy:.4f}"
    )

    print(
        f"TF-IDF + Logistic Regression Macro F1: "
        f"{tfidf_macro_f1:.4f}"
    )

    print()

    print(
        f"Accuracy improvement: "
        f"{accuracy_improvement:+.4f}"
    )

    print(
        f"Macro F1 improvement: "
        f"{macro_f1_improvement:+.4f}"
    )

    print()
    print(
        f"Comparison saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()