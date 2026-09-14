from retrieval import retrieve_similar_cases
from classifier import IntentClassifier
from reply_generator import generate_reply


def decide_escalation(
    customer_message,
    intent,
    confidence,
    evidence
):
    """
    Decide whether the customer message should be
    auto-handled or escalated to a human agent.

    Escalation is triggered by:
    1. High-risk intents
    2. Strong complaint / legal / fraud signals
    3. Compensation or refund demands
    4. Very low classifier confidence combined
       with weak historical evidence

    Strong historical evidence can allow a low-confidence
    routine request to be auto-handled.
    """

    text = customer_message.lower()

    # ---------------------------------------------------------
    # 1. HIGH-RISK INTENTS
    # ---------------------------------------------------------

    if intent == "account_access":
        return (
            "escalate",
            "Account or security-related issue requires human review."
        )

    if intent == "payment_billing":
        return (
            "escalate",
            "Payment or billing-related issue requires human review."
        )

    # ---------------------------------------------------------
    # 2. STRONG COMPLAINT / RISK SIGNALS
    # ---------------------------------------------------------

    complaint_words = [
        "fraud",
        "scam",
        "cheat",
        "lawsuit",
        "legal",
        "sue",
        "terrible",
        "unacceptable",
        "angry",
        "horrible",
        "pathetic",
        "incompetent",
        "worst",
        "disappointed",
        "ridiculous",
    ]

    if any(word in text for word in complaint_words):
        return (
            "escalate",
            "Customer message contains strong complaint or risk signals."
        )

    # ---------------------------------------------------------
    # 3. COMPENSATION / REFUND DEMANDS
    # ---------------------------------------------------------

    compensation_terms = [
        "compensation",
        "reimbursement",
        "refund",
        "money back",
        "give me my money",
        "credit my account",
        "pay me",
    ]

    if any(term in text for term in compensation_terms):
        return (
            "escalate",
            "Customer is requesting financial compensation or reimbursement."
        )

    # ---------------------------------------------------------
    # 4. HISTORICAL EVIDENCE STRENGTH
    # ---------------------------------------------------------

    if evidence is not None and len(evidence) > 0:
        top_similarity = float(
            evidence["similarity"].max()
        )
    else:
        top_similarity = 0.0

    # ---------------------------------------------------------
    # 5. VERY LOW CONFIDENCE + WEAK EVIDENCE
    # ---------------------------------------------------------

    if confidence < 0.20 and top_similarity < 0.35:
        return (
            "escalate",
            "Both intent confidence and historical evidence strength are low."
        )

    # ---------------------------------------------------------
    # 6. LOW CONFIDENCE + STRONG HISTORICAL EVIDENCE
    # ---------------------------------------------------------

    if confidence < 0.40 and top_similarity >= 0.50:
        return (
            "auto_handle",
            "Historical support cases are strongly similar despite low "
            "intent-classification confidence."
        )

    # ---------------------------------------------------------
    # 7. SUFFICIENT CLASSIFICATION CONFIDENCE
    # ---------------------------------------------------------

    if confidence >= 0.40:
        return (
            "auto_handle",
            "Routine support issue with sufficient classification confidence."
        )

    # ---------------------------------------------------------
    # 8. UNCERTAIN CASES
    # ---------------------------------------------------------

    return (
        "escalate",
        "Intent confidence is low and historical evidence is not strong "
        "enough to safely auto-handle the request."
    )


def main():

    customer_message = (
        "My package was supposed to arrive yesterday "
        "but it still hasn't arrived."
    )

    print("CUSTOMER MESSAGE:")
    print(customer_message)

    # ---------------------------------------------------------
    # 1. CLASSIFY INTENT
    # ---------------------------------------------------------

    classifier = IntentClassifier()

    intent, confidence = classifier.predict(
        customer_message
    )

    print("\nINTENT")
    print("=" * 70)
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence:.3f}")

    # ---------------------------------------------------------
    # 2. RETRIEVE HISTORICAL EVIDENCE
    # ---------------------------------------------------------

    evidence = retrieve_similar_cases(
        customer_message,
        top_k=3
    )

    print("\nHISTORICAL EVIDENCE")
    print("=" * 70)

    for i, (_, row) in enumerate(
        evidence.iterrows(),
        start=1
    ):

        print(f"\nCase {i}")
        print(f"Similarity: {row['similarity']:.3f}")

        print("Customer:")
        print(row["customer_text"])

        print("\nAmazon:")
        print(row["amazon_text"])

        print("-" * 70)

    # ---------------------------------------------------------
    # 3. ESCALATION DECISION
    # ---------------------------------------------------------

    decision, reason = decide_escalation(
        customer_message,
        intent,
        confidence,
        evidence
    )

    print("\nESCALATION DECISION")
    print("=" * 70)
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")

    # ---------------------------------------------------------
    # 4. GENERATE GROUNDED REPLY
    # ---------------------------------------------------------

    reply = generate_reply(
        customer_message,
        intent,
        evidence
    )

    print("\nAGENT REPLY")
    print("=" * 70)
    print(reply)


if __name__ == "__main__":
    main()