import pandas as pd

from llm_classifier import classify_with_llm


GOLDEN_FILE = "evaluation/golden_set_200_verified.xlsx"


def main():

    print("Loading Golden Set...")

    df = pd.read_excel(
        GOLDEN_FILE,
        sheet_name="Golden Set"
    )

    df = df.dropna(
        subset=["customer_text", "intent"]
    )

    # Only test 20 examples first
    sample = df.head(20)

    correct = 0

    print("\nTesting 20 examples...")
    print("=" * 70)

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        customer_message = str(
            row["customer_text"]
        )

        actual = row["intent"]

        try:

            predicted, reason = classify_with_llm(
                customer_message
            )

            is_correct = predicted == actual

            if is_correct:
                correct += 1

            print(f"\nExample {i}")
            print(f"Actual:    {actual}")
            print(f"Predicted: {predicted}")
            print(f"Correct:   {is_correct}")
            print(f"Reason:    {reason}")

        except Exception as e:

            print(f"\nExample {i}")
            print(f"ERROR: {e}")

    print("\n")
    print("=" * 70)
    print("LLM PILOT RESULTS")
    print("=" * 70)

    print(f"Correct: {correct}/20")
    print(f"Accuracy: {correct / 20:.2%}")


if __name__ == "__main__":
    main()