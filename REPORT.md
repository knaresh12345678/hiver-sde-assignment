# Hiver SDE Intern Take-Home Assignment
## AI Customer Support Agent — Amazon

### 1. Problem Framing

The goal of this project is to build an AI-assisted customer support agent that can classify incoming customer messages, use historical support interactions as evidence for drafting a response, and decide whether the request should be automatically handled or escalated.

I selected Amazon from the Kaggle Customer Support on Twitter dataset because it provides a large and diverse set of customer-support interactions covering delivery, returns, payments, products, accounts, technical problems, promotions, and general support.

The system focuses on three decisions:

1. **What is the customer's intent?**
2. **What did Amazon historically do in similar cases?**
3. **Can this request be safely auto-handled, or should it be escalated?**

---

## 2. Data and Intent Taxonomy

The dataset contains customer and support messages linked through Twitter response identifiers. I treated `inbound=True` as customer-authored messages and paired them with Amazon support responses using the available response-ID relationships.

The Amazon subset was filtered to English interactions, resulting in **124,211 English customer-support pairs** used for retrieval.

A compact taxonomy of **10 operational intents** was defined from the observed support interactions:

| Intent | Description |
|---|---|
| `delivery_delay` | Package is late or missed the expected delivery date |
| `delivery_missing` | Package is marked delivered, lost, or not received |
| `order_status` | Order, shipping, dispatch, or tracking status |
| `return_refund` | Returns, refunds, replacements, or reimbursement |
| `product_issue` | Damaged, defective, wrong, incomplete, or poorly packaged product |
| `payment_billing` | Payments, unexpected charges, billing, cashback, or balances |
| `account_access` | Login, password, security, locked, or account-access problems |
| `technical_support` | Kindle, Echo, Fire TV, Prime Video, app, or device problems |
| `seller_promotion` | Seller, pricing, discounts, promotions, offers, or shopping questions |
| `customer_service_other` | General complaints, contact requests, praise, or unclear issues |

A **300-example labelled training set** and a **200-example Golden Evaluation Set** were created. The Golden Set used `random_state=42` for reproducible sampling. Labels were produced as AI-assisted candidates and then reviewed and verified.

---

## 3. System Approach

The system uses a lightweight CPU-friendly architecture:

```text
Customer Message
       |
       v
TF-IDF + Logistic Regression
       |
       v
Predicted Intent + Confidence
       |
       +--------------------+
       |                    |
       v                    v
Historical Retrieval    Escalation Logic
       |                    |
       v                    v
Similar Amazon Cases    Auto-handle / Escalate
       |
       v
Grounded Reply Generation
       |
       v
Gemini (optional)
       |
       v
Deterministic fallback if unavailable