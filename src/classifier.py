import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")


class FakeNewsClassifier:
    """
    A multi-model fake news detection system.

    Trains several classifiers on TF-IDF features, evaluates them
    using proper metrics (F1, ROC-AUC), and saves the best model.
    """

    MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    def __init__(self, max_features: int = 10_000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,       # dampens term frequency scores
        )
        self.results: dict = {}
        self.best_model_name: str = ""
        self.best_pipeline = None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate a CSV file with 'text' and 'label' columns."""
        df = pd.read_csv(filepath)
        required = {"text", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required}")
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str).str.lower().str.strip()
        print(f"[data]  Loaded {len(df)} rows | "
              f"Fake: {(df['label']==1).sum()} | Real: {(df['label']==0).sum()}")
        return df

    def split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Stratified train/test split to preserve class balance."""
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"],
            test_size=test_size,
            random_state=42,
            stratify=df["label"],
        )
        print(f"[split] Train: {len(X_train)} | Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def _build_pipeline(self, model) -> Pipeline:
        """Wrap vectorizer + model into a single sklearn Pipeline."""
        return Pipeline([
            ("tfidf",      TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words="english",
                sublinear_tf=True,
            )),
            ("classifier", model),
        ])

    def train_all(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Train every model in MODELS, evaluate on the held-out test set,
        and record F1, ROC-AUC, and accuracy.
        """
        for name, model in self.MODELS.items():
            print(f"\n[train] {name} ...")
            pipe = self._build_pipeline(model)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # ROC-AUC requires probability estimates
            if hasattr(pipe.named_steps["classifier"], "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = None

            report = classification_report(y_test, y_pred, output_dict=True)
            self.results[name] = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "f1_macro":  report["macro avg"]["f1-score"],
                "roc_auc":   auc,
                "pipeline":  pipe,
                "report":    report,
                "confusion":  confusion_matrix(y_test, y_pred),
            }
            auc_str = f"{auc:.3f}" if auc is not None else "n/a"
            print(f"       Accuracy {self.results[name]['accuracy']:.3f} | "
                  f"F1 {self.results[name]['f1_macro']:.3f} | "
                  f"ROC-AUC {auc_str}")

        return self.results

    def select_best(self) -> str:
        """Pick the model with the highest macro F1 score."""
        self.best_model_name = max(
            self.results, key=lambda n: self.results[n]["f1_macro"]
        )
        self.best_pipeline = self.results[self.best_model_name]["pipeline"]
        print(f"\n[best]  {self.best_model_name} "
              f"(F1={self.results[self.best_model_name]['f1_macro']:.3f})")
        return self.best_model_name

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/best_model.pkl"):
        """Pickle the best pipeline to disk for later use."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_name": self.best_model_name,
                "pipeline":   self.best_pipeline,
            }, f)
        print(f"[save]  Model saved → {path}")

    @staticmethod
    def load(path: str = "models/best_model.pkl") -> "FakeNewsClassifier":
        """Load a saved pipeline and return a ready-to-use classifier."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        clf = FakeNewsClassifier()
        clf.best_model_name = data["model_name"]
        clf.best_pipeline   = data["pipeline"]
        return clf

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of text strings.

        Returns a list of dicts with 'label', 'confidence', and 'verdict'.
        """
        if self.best_pipeline is None:
            raise RuntimeError("No model loaded. Call train_all() or load() first.")
        predictions = self.best_pipeline.predict(texts)
        probabilities = self.best_pipeline.predict_proba(texts)

        results = []
        for text, pred, prob in zip(texts, predictions, probabilities):
            confidence = max(prob)
            results.append({
                "text":       text[:80] + "..." if len(text) > 80 else text,
                "label":      int(pred),
                "verdict":    "FAKE" if pred == 1 else "REAL",
                "confidence": round(float(confidence), 4),
            })
        return results
