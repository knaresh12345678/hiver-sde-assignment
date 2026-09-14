import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twitter/twcs/twcs.csv"
CUSTOMER_PATH = "data/processed/amazon_conversations.csv"
OUTPUT_PATH = "data/processed/amazon_pairs.csv"

BRAND = "amazonhelp"

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Building Amazon customer -> support pairs")
print("=" * 60)

# ---------------------------------------------------------
# Load customer tweets that Amazon replied to
# ---------------------------------------------------------

customers = pd.read_csv(
    CUSTOMER_PATH,
    dtype={
        "tweet_id": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
)

customer_ids = set(
    customers["tweet_id"]
    .dropna()
    .astype(str)
)

print(f"Customer messages: {len(customers):,}")

# ---------------------------------------------------------
# Find Amazon replies to those customers
# ---------------------------------------------------------

amazon_replies = []

print("\nScanning dataset for Amazon replies...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ],
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "in_response_to_tweet_id": "string",
    },
    chunksize=100_000,
):
    matches = chunk[
        (chunk["author_id"].str.lower() == BRAND)
        & (
            chunk["in_response_to_tweet_id"]
            .isin(customer_ids)
        )
    ]

    if not matches.empty:
        amazon_replies.append(matches)

if not amazon_replies:
    raise RuntimeError("No Amazon replies found.")

amazon = pd.concat(
    amazon_replies,
    ignore_index=True
)

print(f"Amazon replies found: {len(amazon):,}")

# ---------------------------------------------------------
# Rename columns before joining
# ---------------------------------------------------------

customers = customers.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "created_at": "customer_created_at",
        "text": "customer_text",
    }
)

amazon = amazon.rename(
    columns={
        "tweet_id": "amazon_tweet_id",
        "created_at": "amazon_created_at",
        "text": "amazon_text",
        "in_response_to_tweet_id": "customer_tweet_id",
    }
)

# Keep only the useful columns

customers = customers[
    [
        "customer_tweet_id",
        "customer_created_at",
        "customer_text",
    ]
]

amazon = amazon[
    [
        "amazon_tweet_id",
        "customer_tweet_id",
        "amazon_created_at",
        "amazon_text",
    ]
]

# ---------------------------------------------------------
# Join customer message with Amazon response
# ---------------------------------------------------------

pairs = customers.merge(
    amazon,
    on="customer_tweet_id",
    how="inner",
)

# Remove empty messages

pairs = pairs.dropna(
    subset=["customer_text", "amazon_text"]
)

# Remove duplicate customer/support pairs

pairs = pairs.drop_duplicates(
    subset=["customer_tweet_id", "amazon_tweet_id"]
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

pairs.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 60)
print("Amazon pair extraction complete")
print("=" * 60)

print(f"Customer -> Amazon pairs: {len(pairs):,}")
print(f"Saved to: {OUTPUT_PATH}")

print()
print("Example pairs:")
print("-" * 60)

for _, row in pairs.head(10).iterrows():
    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nAMAZON:")
    print(row["amazon_text"])

    print("-" * 60)