import pandas as pd
from langdetect import detect, DetectorFactory

# Make language detection reproducible
DetectorFactory.seed = 0

INPUT_FILE = "data/processed/amazon_pairs.csv"
OUTPUT_FILE = "data/processed/amazon_english_pairs.csv"


def is_english(text):
    if not isinstance(text, str):
        return False

    text = text.strip()

    # Avoid unreliable detection on extremely short messages
    if len(text) < 20:
        return False

    try:
        return detect(text) == "en"
    except Exception:
        return False


print("Loading Amazon pairs...")
df = pd.read_csv(INPUT_FILE)

print(f"Total Amazon pairs: {len(df):,}")
print("Detecting language...")

df["is_english"] = df["customer_text"].apply(is_english)

english_df = df[df["is_english"]].drop(columns=["is_english"])

print(f"English pairs: {len(english_df):,}")

english_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")