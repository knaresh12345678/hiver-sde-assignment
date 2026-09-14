from retrieval import retrieve_similar_cases
from classifier import IntentClassifier
from reply_generator import generate_reply


def main():

    customer_message = (
        "My package was supposed to arrive yesterday "
        "but it still hasn't arrived."
    )

    print("CUSTOMER MESSAGE")
    print("=" * 70)
    print(customer_message)

    # ---------------------------------------------------------
    # 1. CLASSIFY
    # ---------------------------------------------------------

    classifier = IntentClassifier()

    intent, confidence = classifier.predict(customer_message)

    print("\nINTENT")
    print("=" * 70)
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence:.3f}")

    # ---------------------------------------------------------
    # 2. RETRIEVE EVIDENCE
    # ---------------------------------------------------------

    evidence = retrieve_similar_cases(
        customer_message,
        top_k=3
    )

    print("\nTOP HISTORICAL CASES")
    print("=" * 70)

    for i, (_, row) in enumerate(evidence.iterrows(), start=1):

        print(f"\nCase {i}")
        print(f"Similarity: {row['similarity']:.3f}")
        print(f"Customer: {row['customer_text']}")
        print(f"Amazon: {row['amazon_text']}")

    # ---------------------------------------------------------
    # 3. GENERATE REPLY
    # ---------------------------------------------------------

    print("\nGENERATING REPLY...")
    
    reply = generate_reply(
        customer_message,
        intent,
        evidence
    )

    print("\nGENERATED REPLY")
    print("=" * 70)
    print(reply)


if __name__ == "__main__":
    main()