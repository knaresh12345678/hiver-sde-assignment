import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_FILE = "data/processed/human_reply_evaluation.xlsx"
JUDGE_FILE = "results/llm_judge_results.csv"
OUTPUT_FILE = "results/judge_human_agreement.csv"


def main():

    print("Loading human and judge evaluations...")

    human = pd.read_excel(HUMAN_FILE)
    judge = pd.read_csv(JUDGE_FILE)

    # ---------------------------------------------------------
    # MERGE BY EVALUATION ID
    # ---------------------------------------------------------

    df = human.merge(
        judge,
        on="eval_id",
        how="inner"
    )

    print(f"Matched examples: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "No matching evaluation IDs were found."
        )

    # ---------------------------------------------------------
    # EVIDENCE / GROUNDEDNESS AGREEMENT
    # ---------------------------------------------------------

    human_groundedness = pd.to_numeric(
        df["human_groundedness"],
        errors="coerce"
    )

    judge_groundedness = pd.to_numeric(
        df["judge_groundedness"],
        errors="coerce"
    )

    valid = (
        human_groundedness.notna()
        & judge_groundedness.notna()
    )

    human_scores = human_groundedness[valid]
    judge_scores = judge_groundedness[valid]

    # ---------------------------------------------------------
    # COHEN'S KAPPA
    # ---------------------------------------------------------

    if len(human_scores) >= 2:

        kappa = cohen_kappa_score(
            human_scores,
            judge_scores
        )

    else:

        kappa = float("nan")

    # ---------------------------------------------------------
    # EXACT AGREEMENT
    # ---------------------------------------------------------

    exact_agreement = (
        human_scores.values
        == judge_scores.values
    ).mean()

    # ---------------------------------------------------------
    # WITHIN-ONE AGREEMENT
    # ---------------------------------------------------------

    within_one = (
        abs(
            human_scores.values
            - judge_scores.values
        ) <= 1
    ).mean()

    # ---------------------------------------------------------
    # SAVE CASE-LEVEL RESULTS
    # ---------------------------------------------------------

    agreement_df = pd.DataFrame({
        "eval_id": df.loc[valid, "eval_id"],
        "human_groundedness": human_scores.values,
        "judge_groundedness": judge_scores.values,
        "absolute_difference": abs(
            human_scores.values
            - judge_scores.values
        )
    })

    import os

    os.makedirs(
        "results",
        exist_ok=True
    )

    agreement_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("JUDGE-HUMAN EVIDENCE AGREEMENT")
    print("=" * 70)

    print(
        f"Valid groundedness ratings: "
        f"{len(human_scores)}"
    )

    print(
        f"Exact agreement: "
        f"{exact_agreement:.2%}"
    )

    print(
        f"Within-one agreement: "
        f"{within_one:.2%}"
    )

    print(
        f"Cohen's kappa: "
        f"{kappa:.4f}"
    )

    print()
    print(
        "Agreement results saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()