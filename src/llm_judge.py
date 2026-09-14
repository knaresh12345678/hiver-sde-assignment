import pandas as pd
import os


INPUT_FILE = "data/processed/human_reply_evaluation.xlsx"
OUTPUT_FILE = "results/llm_judge_results.csv"


def evaluate_reply(row):
    """
    Local rule-based judge used when an LLM API is unavailable.

    This is NOT presented as an LLM judge.
    It provides a deterministic baseline for reply evaluation.
    """

    customer = str(row["customer_text"]).lower()
    reply = str(row["generated_reply"]).lower()
    evidence = str(row["historical_evidence"]).lower()

    # ---------------------------------------------------------
    # HALLUCINATION CHECK
    # ---------------------------------------------------------

    hallucination_score = 5

    risky_terms = [
        "guarantee",
        "definitely",
        "your refund is",
        "your refund will",
        "we have issued",
        "we have processed",
        "will arrive on",
        "will arrive by"
    ]

    for term in risky_terms:
        if term in reply:
            hallucination_score = 3
            break

    # ---------------------------------------------------------
    # GROUNDEDNESS
    # ---------------------------------------------------------

    groundedness_score = 2

    evidence_actions = [
        "contact",
        "support",
        "team",
        "provide",
        "details",
        "tracking",
        "order",
        "help",
        "direct message",
        "link"
    ]

    matching_actions = sum(
        1
        for term in evidence_actions
        if term in reply and term in evidence
    )

    if matching_actions >= 3:
        groundedness_score = 4
    elif matching_actions >= 1:
        groundedness_score = 3

    # ---------------------------------------------------------
    # HELPFULNESS
    # ---------------------------------------------------------

    helpfulness_score = 2

    helpful_terms = [
        "please",
        "contact",
        "provide",
        "send",
        "check",
        "tracking",
        "order number",
        "details"
    ]

    helpful_matches = sum(
        1
        for term in helpful_terms
        if term in reply
    )

    if helpful_matches >= 3:
        helpfulness_score = 4
    elif helpful_matches >= 1:
        helpfulness_score = 3

    # ---------------------------------------------------------
    # CORRECTNESS
    # ---------------------------------------------------------

    correctness_score = 2

    intent = str(row["predicted_intent"]).lower()

    intent_terms = {
        "delivery_delay": [
            "delivery",
            "arrive",
            "package",
            "order"
        ],
        "delivery_missing": [
            "package",
            "delivery",
            "order"
        ],
        "return_refund": [
            "return",
            "refund",
            "item"
        ],
        "payment_billing": [
            "payment",
            "charge",
            "account",
            "billing"
        ],
        "account_access": [
            "account",
            "login",
            "password"
        ],
        "technical_support": [
            "device",
            "support",
            "issue",
            "problem"
        ],
    }

    relevant_terms = intent_terms.get(intent, [])

    customer_match = sum(
        1
        for term in relevant_terms
        if term in customer
    )

    reply_match = sum(
        1
        for term in relevant_terms
        if term in reply
    )

    if customer_match >= 1 and reply_match >= 1:
        correctness_score = 4
    elif reply_match >= 1:
        correctness_score = 3

    # ---------------------------------------------------------
    # OVERALL
    # ---------------------------------------------------------

    overall_score = round(
        (
            correctness_score
            + helpfulness_score
            + groundedness_score
            + hallucination_score
        ) / 4
    )

    return {
        "judge_correctness": correctness_score,
        "judge_helpfulness": helpfulness_score,
        "judge_groundedness": groundedness_score,
        "judge_no_hallucination": hallucination_score,
        "judge_overall": overall_score,
        "judge_method": "deterministic_local_baseline"
    }


def main():

    print("Loading human evaluation dataset...")

    df = pd.read_excel(INPUT_FILE)

    print(f"Examples loaded: {len(df)}")

    results = []

    for _, row in df.iterrows():

        scores = evaluate_reply(row)

        result = {
            "eval_id": row["eval_id"],
            **scores
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("LOCAL JUDGE RESULTS")
    print("=" * 70)

    print(
        f"Examples evaluated: {len(results_df)}"
    )

    print()
    print("Average scores:")

    score_columns = [
        "judge_correctness",
        "judge_helpfulness",
        "judge_groundedness",
        "judge_no_hallucination",
        "judge_overall"
    ]

    print(
        results_df[score_columns]
        .mean()
        .round(2)
    )

    print()
    print(
        "Judge method: deterministic local baseline"
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()