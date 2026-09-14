import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


INPUT_FILE = "evaluation/amazon_labeling_300_completed.xlsx"


class IntentClassifier:
    def __init__(self):
        print("Loading labeled examples...")

        df = pd.read_excel(INPUT_FILE)
        df = df.dropna(subset=["customer_text", "intent"])

        X = df["customer_text"].astype(str)
        y = df["intent"].astype(str)

        self.model = Pipeline([
            (
                "features",
                FeatureUnion([
                    (
                        "word",
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 2),
                            min_df=1,
                            max_df=0.98,
                            sublinear_tf=True
                        )
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=1,
                            max_features=30000,
                            sublinear_tf=True
                        )
                    )
                ])
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ])

        print(f"Training on {len(df)} labeled examples...")
        self.model.fit(X, y)

        print("Intent classifier ready.")

    def predict(self, message):
        probabilities = self.model.predict_proba([message])[0]
        classes = self.model.classes_

        best_index = probabilities.argmax()

        intent = classes[best_index]
        confidence = probabilities[best_index]

        return intent, confidence