import os
import json

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)


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


def classify_with_llm(customer_message):

    intent_list = "\n".join(
        f"- {intent}" for intent in INTENTS
    )

    prompt = f"""
You are classifying Amazon customer-support messages.

Choose exactly ONE intent from this list:

{intent_list}

Intent definitions:

delivery_delay:
Package is late or past its expected delivery date.

delivery_missing:
Package is marked delivered/lost/not received.

order_status:
Customer asks about order, shipping, dispatch, or tracking status
without clearly reporting a missed delivery.

return_refund:
Return, refund, replacement, reimbursement, or money-back request.

product_issue:
Product is damaged, defective, incorrect, incomplete, or poor quality.

payment_billing:
Payment, unexpected charge, billing, gift balance, or transaction issue.

account_access:
Login, password, hacked, locked, or account-security issue.

technical_support:
Kindle, Echo, Fire TV, Prime Video, app, or device technical issue.

seller_promotion:
Seller, pricing, discount, promotion, offer, coupon, or shopping issue.

customer_service_other:
General complaint, feedback, praise, contact request, or issue that
does not clearly fit the other categories.

Customer message:
{customer_message}

Important:
- Choose only one intent.
- Base the decision on the customer's actual problem.
- Do not infer a delivery problem just because words like "Amazon"
  or "order" appear.
- If a message clearly describes a missed delivery date, use
  delivery_delay.
- If a package is explicitly marked delivered but not received,
  use delivery_missing.

Return ONLY valid JSON in this format:

{{
  "intent": "one_of_the_allowed_intents",
  "reason": "short explanation"
}}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    text = text.replace("```json", "").replace("```", "").strip()

    result = json.loads(text)

    intent = result["intent"]
    reason = result["reason"]

    if intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned by Gemini: {intent}"
        )

    return intent, reason


if __name__ == "__main__":

    message = (
        "My package was supposed to arrive yesterday "
        "but it still hasn't arrived."
    )

    intent, reason = classify_with_llm(message)

    print("\nCustomer:")
    print(message)

    print("\nGemini intent:")
    print(intent)

    print("\nReason:")
    print(reason)