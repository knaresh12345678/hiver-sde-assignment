import pandas as pd
from collections import Counter

DATA_PATH = "data/raw/twitter/twcs/twcs.csv"
BRAND = "amazonhelp"

print("=" * 60)
print("Analyzing Amazon customer-support data")
print("=" * 60)

brand_messages = 0
customer_messages = 0
customer_texts = []

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "author_id",
        "inbound",
        "text",
    ],
    chunksize=100_000,
):
    # Amazon support messages
    amazon = chunk[
        chunk["author_id"].astype(str).str.lower() == BRAND
    ]

    brand_messages += len(amazon)

    # Customer messages
    customers = chunk[chunk["inbound"] == True]

    customer_messages += len(customers)

    # Keep a limited number of customer messages for analysis
    if len(customer_texts) < 20_000:
        remaining = 20_000 - len(customer_texts)
        customer_texts.extend(
            customers["text"].dropna().astype(str).head(remaining).tolist()
        )

print()
print(f"Amazon support messages: {brand_messages:,}")
print(f"Customer messages scanned: {customer_messages:,}")

# Find common words
word_counts = Counter()

for text in customer_texts:
    words = text.lower().split()

    for word in words:
        word = word.strip(".,!?;:\"'()[]{}")
        
        if len(word) >= 4:
            word_counts[word] += 1

print()
print("Common words in customer messages:")
print("-" * 40)

shown = 0

for word, count in word_counts.most_common(50):
    # Ignore common Twitter words
    if word.startswith("@"):
        continue

    print(f"{word:<25} {count:,}")
    shown += 1

    if shown == 30:
        break

print()
print("Analysis complete.")