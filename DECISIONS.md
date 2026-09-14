# Engineering Decision Log

## 1. Selected Amazon as the target brand

I selected Amazon because it has a large number of customer-support interactions in the dataset and provides enough variety to study delivery, returns, payments, technical issues, accounts, and general support.

## 2. Used inbound customer messages as the primary input

I treated `inbound=True` as a customer message and Amazon's corresponding `inbound=False` messages as support responses. This matches the conversational structure of the dataset.

## 3. Built customer-support pairs from response IDs

I connected customer messages to Amazon responses using `response_tweet_id` and `in_response_to_tweet_id`. This preserves the historical customer-to-support relationship needed for retrieval.

## 4. Filtered the dataset to English interactions

The raw Amazon subset contains multiple languages. I used language detection to create an English-only evaluation and retrieval corpus so that the intent taxonomy and generated responses remain consistent.

## 5. Defined a compact 10-intent taxonomy

Instead of creating many fine-grained categories, I used ten operational intents:

- `delivery_delay`
- `delivery_missing`
- `order_status`
- `return_refund`
- `product_issue`
- `payment_billing`
- `account_access`
- `technical_support`
- `seller_promotion`
- `customer_service_other`

The taxonomy was designed around recurring support problems visible in the sampled Amazon interactions.

## 6. Used AI-assisted candidate labeling followed by verification

The initial labels for the 300-example training set and 200-example Golden Set were generated as candidate labels and then reviewed and verified. This reduced manual labeling effort while ensuring the final Golden Set labels were reviewed.

## 7. Used a 200-example Golden Evaluation Set

I sampled 200 English Amazon interactions using a fixed random seed (`random_state=42`) to create a reproducible evaluation set.

## 8. Used Macro F1 as the primary classification metric

The Golden Set is highly imbalanced, with `customer_service_other` representing a large proportion of examples. Accuracy can therefore be inflated by predicting the dominant class. Macro F1 gives every intent equal weight.

## 9. Chose TF-IDF + Logistic Regression as the main classifier

The classifier uses word-level TF-IDF features with unigram and bigram features followed by balanced Logistic Regression. This provides a lightweight model that runs quickly on CPU-only hardware.

## 10. Added a majority-class baseline

A majority-class classifier was implemented to establish a simple baseline. It predicts the most frequent training intent for every message.

## 11. Used TF-IDF retrieval for historical evidence

Historical customer messages are indexed using TF-IDF. For each new message, the system retrieves similar historical customer cases and uses their corresponding Amazon responses as evidence.

## 12. Prevented retrieval leakage during Golden Set evaluation

All Golden Set customer messages are excluded from the retrieval corpus during evaluation. This prevents the evaluator from retrieving the exact test message and artificially producing a similarity score of 1.0.

## 13. Used evidence similarity as an escalation signal

The escalation policy considers both classifier confidence and historical evidence strength. Strong historical evidence can support automatic handling, while weak evidence combined with low confidence leads to escalation.

## 14. Escalated high-risk support categories

Account/security issues, payment/billing issues, compensation requests, and strong complaint or risk signals are escalated rather than automatically handled. This prioritizes safety and reduces the chance of an inappropriate automated response.

## 15. Used Gemini only for optional response generation

Google Gemini was selected because its free API tier was suitable for experimentation. The response generator is grounded using retrieved historical Amazon cases and is instructed not to invent order details, refund amounts, dates, policies, or actions.

When the Gemini quota was exhausted, the system used a deterministic fallback response instead of failing the complete agent pipeline.