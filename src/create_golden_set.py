import pandas as pd


INPUT_FILE = "data/processed/amazon_english_pairs.csv"
OUTPUT_FILE = "data/processed/golden_set_200.csv"

SAMPLE_SIZE = 200
RANDOM_STATE = 42


# Our 10 intent categories
INTENTS = [
    "delivery_delay",
    "delivery_missing",
    "order_status",
    "return_refund",
    "product_issue",
    "payment_billing",
    "account_access",
    "technical_support",
    "seller_promotion",
    "customer_service_other",
]


print("Loading Amazon English interactions...")

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["customer_text", "amazon_text"])

print(f"Available interactions: {len(df):,}")


# ---------------------------------------------------------
# Create a random sample
# ---------------------------------------------------------

golden = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()


# ---------------------------------------------------------
# Create columns for human annotation
# ---------------------------------------------------------

golden.insert(0, "example_id", range(1, len(golden) + 1))

golden["intent"] = ""
golden["human_notes"] = ""


# Keep only the columns needed for annotation
golden = golden[
    [
        "example_id",
        "customer_text",
        "amazon_text",
        "intent",
        "human_notes",
    ]
]


golden.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\nGolden Evaluation Set created!")
print(f"Examples: {len(golden)}")
print(f"Saved to: {OUTPUT_FILE}")