import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from improved_classifier import ImprovedIntentClassifier


GOLDEN_FILE = "evaluation/golden_set_200_verified.xlsx"
OUTPUT_FILE = "results/improved_golden_predictions.csv"


def main():

    print("Loading Golden Evaluation Set...")

    df = pd.read_excel(
        GOLDEN_FILE,
        sheet_name="Golden Set"
    )

    df = df.dropna(
        subset=["customer_text", "intent"]
    )

    print(f"Golden examples: {len(df)}")

    # ---------------------------------------------------------
    # Train improved classifier
    # ---------------------------------------------------------

    classifier = ImprovedIntentClassifier()

    predictions = []
    confidences = []

    print("\nRunning improved predictions...")

    for i, message in enumerate(
        df["customer_text"].astype(str),
        start=1
    ):

        intent, confidence = classifier.predict(message)

        predictions.append(intent)
        confidences.append(confidence)

        if i % 25 == 0:
            print(f"Processed {i}/{len(df)}")

    df["predicted_intent"] = predictions
    df["confidence"] = confidences

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    y_true = df["intent"]
    y_pred = df["predicted_intent"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n")
    print("=" * 70)
    print("IMPROVED GOLDEN SET RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    # ---------------------------------------------------------
    # Per-intent results
    # ---------------------------------------------------------

    print("\nPER-INTENT RESULTS")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nCONFUSION MATRIX")
    print("=" * 70)
    print(confusion_df)

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPredictions saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()