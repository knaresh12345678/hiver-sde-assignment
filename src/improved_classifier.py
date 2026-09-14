import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


INPUT_FILE = "evaluation/amazon_labeling_300_completed.xlsx"


class ImprovedIntentClassifier:

    def __init__(self):

        print("Loading labeled examples...")

        df = pd.read_excel(INPUT_FILE)

        df = df.dropna(
            subset=["customer_text", "intent"]
        )

        X = df["customer_text"].astype(str)
        y = df["intent"].astype(str)

        # Word features capture meaningful phrases.
        word_features = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True
        )

        # Character features help with Twitter-style text,
        # spelling variations and short phrases.
        char_features = TfidfVectorizer(
            lowercase=True,
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60000,
            sublinear_tf=True
        )

        features = FeatureUnion([
            ("word", word_features),
            ("char", char_features)
        ])

        self.model = Pipeline([
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced"
                )
            )
        ])

        print(
            f"Training on {len(df)} labeled examples..."
        )

        self.model.fit(X, y)

        print("Improved intent classifier ready.")

    def predict(self, message):

        intent = self.model.predict([message])[0]

        probabilities = self.model.predict_proba(
            [message]
        )[0]

        confidence = max(probabilities)

        return intent, confidence


if __name__ == "__main__":

    classifier = ImprovedIntentClassifier()

    message = (
        "My package was supposed to arrive yesterday "
        "but it still hasn't arrived."
    )

    intent, confidence = classifier.predict(message)

    print("\nTest message:")
    print(message)

    print(f"\nPredicted intent: {intent}")
    print(f"Confidence: {confidence:.3f}")