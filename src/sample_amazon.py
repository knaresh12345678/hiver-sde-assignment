import pandas as pd

INPUT_FILE = "data/processed/amazon_sample_300.csv"
OUTPUT_FILE = "data/processed/amazon_labeling_300.csv"

df = pd.read_csv(INPUT_FILE)

# Create a simple labeling template
labeling_df = pd.DataFrame({
    "id": range(1, len(df) + 1),
    "customer_text": df["customer_text"],
    "intent": "",
    "notes": ""
})

labeling_df.to_csv(OUTPUT_FILE, index=False)

print(f"Created labeling file with {len(labeling_df)} messages")
print(f"Saved to: {OUTPUT_FILE}")