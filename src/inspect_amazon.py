import pandas as pd
import random
import re

DATA_PATH = "data/processed/amazon_pairs.csv"

print("=" * 70)
print("Inspecting Amazon customer-support conversations")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nTotal customer-support pairs: {len(df):,}")

# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)

df["amazon_text"] = (
    df["amazon_text"]
    .fillna("")
    .astype(str)
)

# Remove very short customer messages
df = df[df["customer_text"].str.len() >= 15]

# Remove obvious non-English text using a simple heuristic.
# This is only for exploration, not our final language detector.
def looks_english(text):
    letters = re.findall(r"[A-Za-z]", text)

    if len(letters) < 10:
        return False

    english_chars = len(letters)

    total_chars = len(
        re.sub(r"\s", "", text)
    )

    if total_chars == 0:
        return False

    ratio = english_chars / total_chars

    return ratio >= 0.70


english = df[
    df["customer_text"].apply(looks_english)
].copy()

print(
    f"English-like conversations: "
    f"{len(english):,}"
)

# ---------------------------------------------------------
# Random sample
# ---------------------------------------------------------

sample_size = min(50, len(english))

sample = english.sample(
    n=sample_size,
    random_state=42
)

print()
print("=" * 70)
print("50 RANDOM ENGLISH CUSTOMER-SUPPORT EXAMPLES")
print("=" * 70)

for i, (_, row) in enumerate(
    sample.iterrows(),
    start=1
):
    print()
    print(f"EXAMPLE {i}")
    print("-" * 70)

    print("CUSTOMER:")
    print(row["customer_text"])

    print()
    print("AMAZON:")
    print(row["amazon_text"])

    print("-" * 70)

# ---------------------------------------------------------
# Save sample for later analysis
# ---------------------------------------------------------

sample.to_csv(
    "data/processed/amazon_english_sample_50.csv",
    index=False
)

print()
print("=" * 70)
print("Saved:")
print("data/processed/amazon_english_sample_50.csv")
print("=" * 70)