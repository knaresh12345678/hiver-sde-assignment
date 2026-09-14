# Hiver SDE Intern Take-Home Assignment

## AI Customer Support Agent

This project implements an AI-assisted customer support agent using historical Amazon customer-support interactions from the Kaggle Customer Support on Twitter dataset.

The system:

1. Classifies customer messages into a compact support-intent taxonomy.
2. Retrieves historically similar Amazon support cases.
3. Drafts a response grounded in historical Amazon responses.
4. Decides whether the request can be auto-handled or should be escalated.
5. Evaluates classification, retrieval, and escalation behaviour on a 200-example Golden Evaluation Set.

---

# 1. Project Structure

```text
hiver-sde-assignment/
|
+-- data/
|   +-- raw/                  # Downloaded dataset; not committed
|   +-- processed/            # Generated datasets; not committed
|
+-- evaluation/
|   +-- amazon_labeling_300_completed.xlsx
|   +-- golden_set_200_verified.xlsx
|
+-- results/
|   +-- agent_evaluation.csv
|   +-- baseline_comparison.csv
|   +-- golden_set_errors.csv
|   +-- golden_set_predictions.csv
|   +-- improved_golden_predictions.csv
|   +-- judge_human_agreement.csv
|   +-- llm_judge_results.csv
|   +-- top_failure_patterns.csv
|
+-- src/
|   +-- agent.py
|   +-- classifier.py
|   +-- retrieval.py
|   +-- reply_generator.py
|   +-- compare_baselines.py
|   +-- evaluate_agent.py
|   +-- failure_analysis.py
|   +-- ...
|
+-- DECISIONS.md
+-- README.md
+-- REPORT.md
+-- requirements.txt
+-- .gitignore