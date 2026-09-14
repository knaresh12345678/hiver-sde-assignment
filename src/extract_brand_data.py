import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twitter/twcs/twcs.csv"
OUTPUT_PATH = "data/processed/amazon_conversations.csv"
BRAND = "amazonhelp"

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Extracting Amazon customer-support conversations")
print("=" * 60)

# ---------------------------------------------------------
# PASS 1
# Find customer tweet IDs that Amazon directly replied to
# ---------------------------------------------------------

amazon_parent_ids = set()
amazon_message_count = 0

print("\nPass 1/2: Finding customer messages answered by Amazon...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "in_response_to_tweet_id",
    ],
    dtype={
        "tweet_id": "string",
        "in_response_to_tweet_id": "string",
        "author_id": "string",
    },
    chunksize=100_000,
):
    amazon_rows = chunk[
        chunk["author_id"].str.lower() == BRAND
    ]

    amazon_message_count += len(amazon_rows)

    parent_ids = amazon_rows[
        "in_response_to_tweet_id"
    ].dropna()

    amazon_parent_ids.update(
        parent_ids.tolist()
    )

print(f"Amazon support messages found: {amazon_message_count:,}")
print(
    f"Customer messages Amazon replied to: "
    f"{len(amazon_parent_ids):,}"
)

# ---------------------------------------------------------
# PASS 2
# Retrieve those customer messages
# ---------------------------------------------------------

customer_rows = []

print("\nPass 2/2: Retrieving customer messages...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "in_response_to_tweet_id": "string",
        "response_tweet_id": "string",
    },
    chunksize=100_000,
):
    matching = chunk[
        chunk["tweet_id"].isin(amazon_parent_ids)
    ]

    if not matching.empty:
        customer_rows.append(matching)

if customer_rows:
    customers = pd.concat(
        customer_rows,
        ignore_index=True
    )
else:
    customers = pd.DataFrame()

print(
    f"Customer messages retrieved: "
    f"{len(customers):,}"
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

customers.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 60)
print("Extraction complete")
print("=" * 60)
print(f"Saved to: {OUTPUT_PATH}")
print(f"Rows saved: {len(customers):,}")