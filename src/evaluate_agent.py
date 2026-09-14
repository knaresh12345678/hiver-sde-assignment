import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from classifier import IntentClassifier
from retrieval import retrieve_similar_cases
from agent import decide_escalation


GOLDEN_FILE = "evaluation/golden_set_200_verified.xlsx"
OUTPUT_FILE = "results/agent_evaluation.csv"


def main():

    print("Loading Golden Evaluation Set...")

    df = pd.read_excel(
        GOLDEN_FILE,
        sheet_name="Golden Set"
    )

    df = df.dropna(
        subset=["customer_text", "intent"]
    ).reset_index(drop=True)

    print(f"Golden examples: {len(df)}")

    # ---------------------------------------------------------
    # CREATE GOLDEN TEXT EXCLUSION SET
    # ---------------------------------------------------------
    # All Golden Set customer messages are excluded from
    # historical retrieval during evaluation.
    # This prevents retrieval data leakage.

    golden_customer_texts = set(
        df["customer_text"].astype(str)
    )

    # ---------------------------------------------------------
    # LOAD CLASSIFIER
    # ---------------------------------------------------------

    classifier = IntentClassifier()

    results = []

    print()
    print("Running full agent evaluation...")
    print("=" * 70)

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        customer_message = str(row["customer_text"])
        actual_intent = str(row["intent"])

        # -----------------------------------------------------
        # 1. INTENT CLASSIFICATION
        # -----------------------------------------------------

        predicted_intent, confidence = classifier.predict(
            customer_message
        )

        # -----------------------------------------------------
        # 2. HISTORICAL RETRIEVAL
        # -----------------------------------------------------
        # Exclude:
        #   - the current Golden Set message
        #   - every other Golden Set message
        #
        # This ensures evaluation does not retrieve any
        # Golden Set example as historical evidence.

        evidence = retrieve_similar_cases(
            customer_message,
            top_k=3,
            exclude_customer_text=customer_message,
            exclude_customer_texts=golden_customer_texts
        )

        if len(evidence) > 0:
            top_similarity = float(
                evidence["similarity"].max()
            )
        else:
            top_similarity = 0.0

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
        # STORE RESULT
        # -----------------------------------------------------

        results.append({
            "example_id": i,
            "customer_text": customer_message,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "top_similarity": top_similarity,
            "escalation_decision": decision,
            "escalation_reason": reason
        })

        if i % 25 == 0:
            print(
                f"Processed {i}/{len(df)}"
            )

    # ---------------------------------------------------------
    # CREATE RESULTS DATAFRAME
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # INTENT METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        results_df["actual_intent"],
        results_df["predicted_intent"]
    )

    macro_f1 = f1_score(
        results_df["actual_intent"],
        results_df["predicted_intent"],
        average="macro"
    )

    # ---------------------------------------------------------
    # RETRIEVAL METRICS
    # ---------------------------------------------------------

    mean_similarity = results_df[
        "top_similarity"
    ].mean()

    strong_evidence_rate = (
        results_df["top_similarity"] >= 0.50
    ).mean()

    # ---------------------------------------------------------
    # ESCALATION METRICS
    # ---------------------------------------------------------

    auto_handle_rate = (
        results_df["escalation_decision"]
        == "auto_handle"
    ).mean()

    escalate_rate = (
        results_df["escalation_decision"]
        == "escalate"
    ).mean()

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("AGENT EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Examples evaluated : {len(results_df)}"
    )

    print(
        f"Intent accuracy    : {accuracy:.4f}"
    )

    print(
        f"Intent Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Mean top similarity: {mean_similarity:.4f}"
    )

    print(
        f"Strong evidence rate: "
        f"{strong_evidence_rate:.2%}"
    )

    print(
        f"Auto-handle rate   : "
        f"{auto_handle_rate:.2%}"
    )

    print(
        f"Escalation rate    : "
        f"{escalate_rate:.2%}"
    )

    print()
    print(
        "Golden Set retrieval leakage prevention: ENABLED"
    )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()