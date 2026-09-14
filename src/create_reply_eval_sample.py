import pandas as pd

from retrieval import retrieve_similar_cases
from classifier import IntentClassifier
from reply_generator import generate_reply
from agent import decide_escalation


GOLDEN_FILE = "evaluation/golden_set_200_verified.xlsx"
OUTPUT_FILE = "data/processed/reply_eval_sample.csv"


def main():

    print("Loading Golden Set...")

    df = pd.read_excel(
        GOLDEN_FILE,
        sheet_name="Golden Set"
    )

    df = df.dropna(
        subset=["customer_text", "intent"]
    )

    # Fixed seed makes the evaluation reproducible.
    sample = df.sample(
        n=10,
        random_state=42
    ).copy()

    classifier = IntentClassifier()

    results = []

    print()
    print("=" * 70)
    print("GENERATING REPLY EVALUATION DATA")
    print("=" * 70)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):

        customer_message = str(row["customer_text"])
        actual_intent = str(row["intent"])

        print()
        print(f"Example {i}/10")
        print(f"Customer: {customer_message}")

        # -----------------------------------------------------
        # 1. CLASSIFY
        # -----------------------------------------------------

        predicted_intent, confidence = classifier.predict(
            customer_message
        )

        # -----------------------------------------------------
        # 2. RETRIEVE HISTORICAL EVIDENCE
        # -----------------------------------------------------

        evidence = retrieve_similar_cases(
            customer_message,
            top_k=3
        )

        # -----------------------------------------------------
        # 3. ESCALATION DECISION
        # -----------------------------------------------------

        decision, reason = decide_escalation(
            customer_message,
            predicted_intent,
            confidence,
            evidence
        )

        # -----------------------------------------------------
        # 4. GENERATE REPLY
        # -----------------------------------------------------

        reply = generate_reply(
            customer_message,
            predicted_intent,
            evidence
        )

        # Store the top three evidence cases.
        evidence_text = ""

        for j, (_, evidence_row) in enumerate(
            evidence.iterrows(),
            start=1
        ):
            evidence_text += (
                f"Case {j} "
                f"(similarity={evidence_row['similarity']:.3f})\n"
                f"Customer: {evidence_row['customer_text']}\n"
                f"Amazon: {evidence_row['amazon_text']}\n\n"
            )

        results.append({
            "eval_id": i,
            "customer_text": customer_message,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "escalation_decision": decision,
            "escalation_reason": reason,
            "historical_evidence": evidence_text.strip(),
            "generated_reply": reply
        })

        print(f"Predicted intent: {predicted_intent}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Decision: {decision}")

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("REPLY EVALUATION DATASET CREATED")
    print("=" * 70)
    print(f"Examples: {len(results_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()