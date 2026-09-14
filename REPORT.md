# Hiver SDE Intern Take-Home Assignment
## AI Customer Support Agent - Amazon

## 1. Problem Framing

The goal is to build an AI-assisted customer support agent that can classify incoming customer messages, retrieve historically similar support interactions, draft a grounded response, and decide whether a request should be automatically handled or escalated.

I selected Amazon from the Kaggle Customer Support on Twitter dataset because it provides a large and diverse set of customer-support interactions covering delivery, returns, payments, products, accounts, technical problems, promotions, and general support.

The system answers three operational questions:

1. **What is the customer's intent?**
2. **What did Amazon historically do in similar cases?**
3. **Can this request be safely auto-handled, or should it be escalated?**

The design prioritizes reproducibility and CPU-friendly methods because the development environment has limited hardware resources.

---

## 2. Data and Intent Taxonomy

The dataset contains customer and support messages linked through Twitter response identifiers. I treated `inbound=True` as customer-authored messages and paired them with Amazon support responses using the available response-ID relationships.

The Amazon subset contained 169,840 support messages and 155,445 customer messages that Amazon replied to. After English-language filtering, **124,211 customer-support pairs** remained for retrieval and evaluation.

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

A **300-example labelled training set** and a **200-example Golden Evaluation Set** were created. The Golden Set was sampled with `random_state=42` for reproducibility. Labels were created as AI-assisted candidates and then reviewed and verified by the project owner.

### Golden Set Sampling and Verification

The 200 Golden examples were randomly sampled from the English Amazon customer-support pairs. Verification was performed after candidate labelling rather than treating model-generated labels as ground truth. This makes the set useful for evaluation while keeping the labelling process transparent.

---

## 3. System Architecture

The system is intentionally lightweight:

```text
Customer Message
       |
       v
TF-IDF + Logistic Regression
       |
       v
Predicted Intent + Confidence
       |
       +---------------------+
       |                     |
       v                     v
Historical Retrieval    Escalation Logic
       |                     |
       v                     v
Similar Amazon Cases   Auto-handle / Escalate
       |
       v
Grounded Reply Generation
       |
       v
Gemini (optional)
       |
       v
Deterministic fallback if unavailable

---

## 4. Evaluation Methodology

The main classification evaluation uses the **200-example Golden Evaluation Set**.

Because the intent distribution is imbalanced, **Macro F1** is treated as the primary metric. Accuracy is reported as a secondary metric because a model can obtain high accuracy by over-predicting the dominant `customer_service_other` class.

Two baselines were evaluated on exactly the same 200 Golden examples:

1. Majority-class baseline
2. TF-IDF + Logistic Regression classifier

The full agent was then evaluated using the same Golden Set, including retrieval and escalation.

---

## 5. Baseline Results

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | **63.50%** | **7.77%** |
| TF-IDF + Logistic Regression | **48.50%** | **45.53%** |
| Full agent | **48.00%** | **45.45%** |

The majority baseline predicts `customer_service_other` for every example. It therefore achieves 63.5% accuracy because that is the dominant class, but its Macro F1 is only 7.77%.

The TF-IDF classifier has lower accuracy but substantially higher Macro F1. This demonstrates that **accuracy alone gives a misleading impression of performance** for this taxonomy.

The full agent produces almost the same intent performance as the standalone classifier because retrieval and escalation operate after intent prediction rather than replacing the classifier.

---

## 6. Agent Evaluation

On the 200-example Golden Set:

- **Intent accuracy:** 48.00%
- **Macro F1:** 45.45%
- **Mean top retrieval similarity:** 0.3542
- **Strong evidence rate (similarity >= 0.50):** 11.00%
- **Auto-handle rate:** 28.50%
- **Escalation rate:** 71.50%

The high escalation rate is deliberate. The agent is designed to be conservative when intent confidence is low, evidence is weak, or a request involves sensitive account, payment, complaint, legal, or compensation-related signals.

---

## 7. Failure Analysis

There were **104 incorrect predictions out of 200 Golden examples**, giving an overall error rate of 52%.

The largest failure source is the broad `customer_service_other` category.

### Top Confusion Patterns

| Actual intent | Predicted intent | Count |
|---|---|---:|
| `customer_service_other` | `delivery_delay` | 37 |
| `customer_service_other` | `delivery_missing` | 14 |
| `customer_service_other` | `payment_billing` | 6 |
| `customer_service_other` | `seller_promotion` | 5 |
| `customer_service_other` | `technical_support` | 5 |

These five patterns account for **67 of the 104 errors (64.4%)**.

### Failure Pattern 1 - Broad "Other" Class

Many customer messages contain enough language to resemble a concrete operational problem, but the verified label is `customer_service_other`. The classifier tends to assign these messages to more specific classes such as delivery.

**Root cause:** `customer_service_other` is semantically broad and contains many different support situations.

**Improvement:** split the broad class into more operational categories and add more verified training examples.

### Failure Pattern 2 - Delivery Delay vs. Missing Package

Messages about late packages and messages about packages that were never received share substantial vocabulary such as "package", "delivery", "arrived", and "received".

**Root cause:** word-level TF-IDF does not reliably distinguish the subtle semantic difference between delayed delivery and confirmed non-receipt.

**Improvement:** add explicit features for temporal language and delivery state, and increase labelled examples for both classes.

### Failure Pattern 3 - Payment vs. Refund

Payment, charges, refund, reimbursement, and balance-related messages overlap.

**Root cause:** several financial concepts occur in both payment and return/refund interactions.

**Improvement:** introduce finer-grained financial labels and explicit rules for charge/payment versus post-return reimbursement.

### Failure Pattern 4 - Weak Historical Evidence

**84 of 200 examples (42%)** had a top retrieval similarity below 0.30.

**Root cause:** TF-IDF similarity depends heavily on shared vocabulary and cannot reliably retrieve semantically similar cases when users phrase the same problem differently.

**Improvement:** replace or augment TF-IDF retrieval with sentence embeddings and evaluate retrieval recall separately.

### Failure Pattern 5 - Conservative Escalation

The system escalated 71.5% of Golden examples.

The main escalation reason was low intent confidence combined with weak historical evidence. This reduces the chance of an unsafe automated answer but also limits automation coverage.

**Improvement:** calibrate classifier confidence and retrieval thresholds on a larger validation set and measure the precision of auto-handled cases separately.

---

## 8. What Is Misleading About My Headline Number?

The most misleading headline number would be the **63.5% accuracy of the majority baseline**.

At first glance, 63.5% appears better than the classifier's 48.5% accuracy. However, the majority baseline simply predicts `customer_service_other` for every message.

Its Macro F1 is only **7.77%**, showing that it performs extremely poorly across the minority intents.

Therefore, the headline classification number should not be "63.5% accuracy." The more informative headline is:

> **TF-IDF + Logistic Regression reaches 45.53% Macro F1 on the 200-example Golden Set, compared with 7.77% for the majority baseline.**

Even this number should be interpreted cautiously because the Golden Set is relatively small, labels were AI-assisted before human verification, and the taxonomy contains a broad `customer_service_other` class.

The evaluation should therefore be viewed as a diagnostic baseline rather than evidence that the system is production-ready.

---

## 9. Reply Evaluation and LLM-as-Judge

A five-dimension rubric was defined for response evaluation:

1. **Correctness** - Does the response address the customer's actual issue?
2. **Helpfulness** - Does it provide a useful next step?
3. **Groundedness** - Is it supported by retrieved historical evidence?
4. **No hallucination** - Does it avoid unsupported details or promises?
5. **Overall quality** - Would the response be acceptable to a customer-support reviewer?

Each dimension uses a 1-5 scale.

A deterministic local judge was used as a diagnostic fallback because the available Gemini free-tier quota was exhausted during the evaluation process. A genuine Gemini LLM-as-judge run could therefore not be completed reliably.

The local diagnostic produced:

| Dimension | Mean |
|---|---:|
| Correctness | 3.0 |
| Helpfulness | 4.0 |
| Groundedness | 2.6 |
| No hallucination | 5.0 |
| Overall | 3.9 |

These values should **not** be presented as genuine LLM-as-judge results.

A separate 10-example human reply-evaluation sheet was prepared using the same rubric. The current ratings are AI-assisted candidate ratings and require independent human verification. Therefore, the calculated agreement statistics are diagnostic only and are not claimed as valid human-vs-LLM agreement.

This limitation is explicitly documented rather than fabricating an LLM judge result.

---

## 10. One-Week Next Steps

If given one additional week, I would prioritize:

### 1. Improve the Intent Dataset

Expand the verified labelled set from 300 examples to approximately 1,000-2,000 examples, focusing on minority classes and the major confusion pairs.

### 2. Redesign the Taxonomy

Split `customer_service_other` into smaller operational intents where the data supports it. This should reduce the largest source of classification errors.

### 3. Upgrade Retrieval

Replace TF-IDF-only retrieval with a lightweight sentence-embedding model and compare retrieval quality using Recall@K and evidence relevance.

### 4. Calibrate Automation

Tune confidence and evidence thresholds using a validation set. The key production metric should be **precision of auto-handled requests**, not simply the percentage of requests automated.

### 5. Strengthen Response Evaluation

Run a genuine LLM-as-judge evaluation when API capacity is available, compare it against independently human-rated examples, and calculate agreement using a frozen evaluation set.

The goal would be to improve automation coverage while keeping incorrect or unsupported automated responses below an explicitly defined safety threshold.

---

## 11. Conclusion

This project demonstrates a complete, reproducible customer-support agent pipeline using a relatively small verified training set and a lightweight CPU-friendly architecture.

The main lesson from the evaluation is that **accuracy alone is not sufficient** for this imbalanced support-intent problem. The majority baseline reaches 63.5% accuracy while essentially failing the minority classes. Macro F1 gives a more informative view of intent coverage.

The largest technical limitation is the broad `customer_service_other` class combined with weak lexical retrieval. The current system therefore uses conservative escalation to reduce the risk of unsupported automated handling.

The next iteration should focus on better-labelled data, a refined taxonomy, semantic retrieval, confidence calibration, and independently verified response evaluation.

Supporting implementation decisions are recorded in `DECISIONS.md`.
