import pandas as pd


INPUT_FILE = "results/golden_set_predictions.csv"
OUTPUT_FILE = "results/golden_set_errors.csv"


def main():

    print("Loading Golden Set predictions...")

    df = pd.read_csv(INPUT_FILE)

    # Keep only incorrect predictions
    errors = df[
        df["intent"] != df["predicted_intent"]
    ].copy()

    print(f"Total examples: {len(df)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(
        f"Error rate: {len(errors) / len(df):.2%}"
    )

    print("\nTOP CONFUSIONS")
    print("=" * 70)

    confusion = (
        errors
        .groupby(["intent", "predicted_intent"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(confusion.to_string(index=False))

    # ---------------------------------------------------------
    # Save detailed errors
    # ---------------------------------------------------------

    errors = errors[
        [
            "example_id",
            "customer_text",
            "intent",
            "predicted_intent",
            "confidence",
            "amazon_text"
        ]
    ]

    errors.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDetailed errors saved to:")
    print(OUTPUT_FILE)

    # ---------------------------------------------------------
    # Show first 20 errors
    # ---------------------------------------------------------

    print("\nFIRST 20 ERRORS")
    print("=" * 70)

    for _, row in errors.head(20).iterrows():

        print(f"\nExample ID: {row['example_id']}")
        print(f"Actual:    {row['intent']}")
        print(f"Predicted: {row['predicted_intent']}")
        print(f"Confidence: {row['confidence']:.3f}")

        print("\nCustomer:")
        print(row["customer_text"])

        print("\nHistorical Amazon response:")
        print(row["amazon_text"])

        print("-" * 70)


if __name__ == "__main__":
    main()