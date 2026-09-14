# Hiver SDE Intern Take-Home Assignment

## AI Customer Support Agent

This project implements an AI-assisted customer support agent using historical Amazon customer-support interactions from the Kaggle Customer Support on Twitter dataset.

The system:

1. Classifies customer messages into a compact support-intent taxonomy.
2. Retrieves historically similar Amazon support cases.
3. Drafts a response grounded in historical Amazon responses.
4. Decides whether the request can be auto-handled or should be escalated.
5. Evaluates classification, retrieval, and escalation behavior on a 200-example Golden Evaluation Set.

---

# 1. Project Structure

```text
hiver-sde-assignment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│
├── notebooks/
│
├── results/
│
├── src/
│   ├── analyze_brand.py
│   ├── extract_brand_data.py
│   ├── build_amazon_pairs.py
│   ├── inspect_amazon.py
│   ├── filter_english.py
│   ├── sample_amazon.py
│   ├── create_golden_set.py
│   ├── classifier.py
│   ├── baseline_classifier.py
│   ├── majority_baseline.py
│   ├── compare_baselines.py
│   ├── retrieval.py
│   ├── agent.py
│   ├── reply_generator.py
│   ├── create_reply_eval_sample.py
│   ├── create_human_eval.py
│   ├── llm_judge.py
│   ├── calculate_agreement.py
│   └── failure_analysis.py
│
├── .gitignore
├── DECISIONS.md
├── README.md
└── requirements.txt