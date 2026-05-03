import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_dataset
from classifier    import FakeNewsClassifier
from visualize     import plot_model_comparison, plot_confusion_matrix, plot_top_features


DEMO_TEXTS = [
    "Scientists from Oxford publish peer-reviewed study linking processed sugar "
    "intake to increased inflammation markers in adults over 50.",

    "SHOCKING: Government is secretly putting mind control chemicals in tap water "
    "to keep you docile. Share this before they delete it!",

    "Central bank releases quarterly GDP data showing moderate 2.3% growth, "
    "in line with analyst expectations.",

    "EXPOSED: 5G towers are bioweapons designed by global elite to depopulate "
    "the planet. Doctors are being silenced RIGHT NOW.",
]


def run_training():
    print("=" * 55)
    print("  Fake News Detector — Training Pipeline")
    print("=" * 55)

    # 1. Generating samples
    data_path = "data/news.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        print("\n[step 1] Generating dataset...")
        generate_dataset(n_samples=3000, output_path=data_path)
    else:
        print(f"\n[step 1] Dataset found at {data_path}")

    # 2. Load + split
    print("\n[step 2] Loading and splitting data...")
    clf = FakeNewsClassifier(max_features=10_000, ngram_range=(1, 2))
    df = clf.load_data(data_path)
    X_train, X_test, y_train, y_test = clf.split(df)

    # 3. Train all models
    print("\n[step 3] Training models...")
    results = clf.train_all(X_train, X_test, y_train, y_test)

    # 4. Select best
    print("\n[step 4] Selecting best model...")
    best = clf.select_best()

    # 5. Save
    print("\n[step 5] Saving best model...")
    clf.save("models/best_model.pkl")

    # 6. Generate plots
    print("\n[step 6] Generating evaluation plots...")
    os.makedirs("results", exist_ok=True)
    plot_model_comparison(results)
    plot_confusion_matrix(results[best]["confusion"], best)
    plot_top_features(results[best]["pipeline"])

    # 7. Demo predictions
    print("\n[step 7] Demo predictions on unseen text:")
    print("-" * 55)
    preds = clf.predict(DEMO_TEXTS)
    for p in preds:
        tag = "🔴 FAKE" if p["verdict"] == "FAKE" else "🟢 REAL"
        print(f"{tag}  ({p['confidence']*100:.1f}% confidence)")
        print(f"   \"{p['text']}\"")
        print()

    print("=" * 55)
    print("  Done. Check results/ for evaluation plots.")
    print("=" * 55)


def run_predict(text: str):
    """Load saved model and predict a single text string."""
    clf = FakeNewsClassifier.load("models/best_model.pkl")
    preds = clf.predict([text])
    p = preds[0]
    print(f"\nVerdict   : {p['verdict']}")
    print(f"Confidence: {p['confidence']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("--predict", type=str, default=None,
                        help="Predict a single text string")
    args = parser.parse_args()

    if args.predict:
        run_predict(args.predict)
    else:
        run_training()
