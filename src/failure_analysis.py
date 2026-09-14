import pandas as pd
from collections import Counter


INPUT_FILE = "results/agent_evaluation.csv"
OUTPUT_FILE = "results/top_failure_patterns.csv"


def main():

    print("Loading evaluation results...")

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # CLASSIFICATION ERRORS
    # ---------------------------------------------------------

    errors = df[
        df["actual_intent"] != df["predicted_intent"]
    ].copy()

    print()
    print("=" * 70)
    print("CLASSIFICATION FAILURE ANALYSIS")
    print("=" * 70)

    print(f"Total examples: {len(df)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(
        f"Error rate: "
        f"{len(errors) / len(df):.2%}"
    )

    # ---------------------------------------------------------
    # TOP CONFUSIONS
    # ---------------------------------------------------------

    confusion_counts = (
        errors
        .groupby(
            ["actual_intent", "predicted_intent"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print()
    print("TOP INTENT CONFUSIONS")
    print("-" * 70)

    for _, row in confusion_counts.head(10).iterrows():

        print(
            f"{row['actual_intent']}"
            f" -> "
            f"{row['predicted_intent']}"
            f": "
            f"{int(row['count'])}"
        )

    # ---------------------------------------------------------
    # LOW-EVIDENCE CASES
    # ---------------------------------------------------------

    low_evidence = df[
        df["top_similarity"] < 0.30
    ]

    print()
    print("LOW-EVIDENCE CASES")
    print("-" * 70)

    print(
        f"Cases with top similarity < 0.30: "
        f"{len(low_evidence)} "
        f"({len(low_evidence) / len(df):.2%})"
    )

    # ---------------------------------------------------------
    # ESCALATION BREAKDOWN
    # ---------------------------------------------------------

    print()
    print("ESCALATION BREAKDOWN")
    print("-" * 70)

    escalation_counts = (
        df["escalation_reason"]
        .value_counts()
    )

    for reason, count in escalation_counts.items():

        print(
            f"{count:3d} | {reason}"
        )

    # ---------------------------------------------------------
    # SAVE TOP CONFUSIONS
    # ---------------------------------------------------------

    confusion_counts.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # SHOW EXAMPLES FROM TOP 5 CONFUSIONS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLES FROM TOP 5 FAILURE PATTERNS")
    print("=" * 70)

    for _, confusion in confusion_counts.head(5).iterrows():

        actual = confusion["actual_intent"]
        predicted = confusion["predicted_intent"]

        matching = errors[
            (errors["actual_intent"] == actual)
            &
            (errors["predicted_intent"] == predicted)
        ]

        print()
        print(
            f"{actual} -> {predicted} "
            f"({len(matching)} cases)"
        )
        print("-" * 70)

        for _, example in matching.head(2).iterrows():

            print(
                f"Customer: {example['customer_text']}"
            )

            print(
                f"Confidence: "
                f"{float(example['confidence']):.3f}"
            )

            print(
                f"Top similarity: "
                f"{float(example['top_similarity']):.3f}"
            )

            print()

    print()
    print(f"Saved confusion analysis to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()