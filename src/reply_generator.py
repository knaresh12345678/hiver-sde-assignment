import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)


def generate_reply(customer_message, intent, evidence):
    historical_cases = ""

    for i, (_, row) in enumerate(evidence.iterrows(), start=1):
        historical_cases += f"""
Case {i}

Customer:
{row["customer_text"]}

Amazon response:
{row["amazon_text"]}

Similarity:
{row["similarity"]:.3f}

---
"""

    prompt = f"""
You are an Amazon customer support assistant.

Your job is to draft a short, professional reply to the customer.

Customer message:
{customer_message}

Detected intent:
{intent}

Historical Amazon support cases:
{historical_cases}

Instructions:
1. Use the historical cases as the primary grounding source.
2. Do not invent order details, refund amounts, delivery dates, policies, or actions.
3. Do not claim that you performed an action.
4. If the historical responses ask the customer to check tracking, provide information,
   or contact support, follow that general pattern.
5. Keep the response concise and helpful.
6. Do not mention that you are an AI.
7. Do not mention the historical cases.

Return only the customer-facing reply.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return (
            "I'm sorry you're having trouble with your order. "
            "Please check the latest tracking information for the most up-to-date status."
        )

    except Exception as e:
        print(f"\nWARNING: Gemini reply generation failed: {e}")

        return (
            "I'm sorry you're experiencing this issue. "
            "Please check the latest tracking information for your order. "
            "If the issue continues, please contact Amazon customer support for further assistance."
        )