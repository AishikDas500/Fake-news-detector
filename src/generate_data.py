"""
generate_data.py
----------------
Generates a synthetic labelled dataset for fake news detection.

In a real project you would swap this with:
  - LIAR dataset (https://huggingface.co/datasets/liar)
  - FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet)
This script lets the project run immediately without downloading anything.
"""
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

REAL_TEMPLATES = [
    "Scientists publish peer-reviewed study showing {topic} linked to {outcome}.",
    "Government officials confirm {topic} after months of investigation.",
    "New research from {university} suggests {topic} may reduce {outcome}.",
    "Health authorities recommend {topic} based on clinical trial data.",
    "{topic} rates decline for third consecutive year, data shows.",
    "Experts caution against overstating {topic} without further evidence.",
    "Independent audit confirms {topic} meets international safety standards.",
    "Study of {n} participants reveals statistically significant link between {topic} and {outcome}.",
    "Central bank releases quarterly report on {topic} citing moderate growth.",
    "Bipartisan committee approves funding for {topic} research initiative.",
]

FAKE_TEMPLATES = [
    "SHOCKING: {topic} CAUSES {outcome} — what the government is hiding!",
    "They don't want you to know this about {topic}. Share before deleted!",
    "{topic} is a HOAX designed to control the population. Wake up!",
    "BREAKING: Secret documents prove {topic} was planned all along.",
    "100% PROVEN: {topic} cures {outcome} instantly — doctors furious!",
    "The TRUTH about {topic} they are desperately suppressing RIGHT NOW.",
    "EXPOSED: {topic} connected to global elite conspiracy. Must watch!",
    "WARNING: {topic} contains hidden {outcome} risks nobody is telling you.",
    "LEAKED memo shows {topic} is completely fabricated by mainstream media.",
    "You will NOT believe what {topic} is really doing to your {outcome}!",
]

TOPICS = [
    "vitamin D supplementation", "electric vehicles", "5G networks",
    "mRNA vaccines", "remote work policies", "artificial intelligence",
    "solar panel efficiency", "water fluoridation", "mask mandates",
    "cryptocurrency regulation", "microplastics", "nuclear energy",
    "gene editing", "social media algorithms", "carbon taxes",
]

OUTCOMES = [
    "cancer risk", "cognitive decline", "immune response",
    "economic growth", "mental health", "birth rates",
    "life expectancy", "fertility rates", "blood pressure",
    "sleep quality",
]

UNIVERSITIES = [
    "MIT", "Stanford", "Johns Hopkins", "Oxford", "Cambridge",
    "Harvard Medical School", "ETH Zurich", "Imperial College",
]


def _fill(template: str) -> str:
    return template.format(
        topic=random.choice(TOPICS),
        outcome=random.choice(OUTCOMES),
        university=random.choice(UNIVERSITIES),
        n=random.randint(500, 50_000),
    )


def generate_dataset(n_samples: int = 2000, output_path: str = "data/news.csv"):
    """
    Generate n_samples rows, balanced 50/50 fake (1) vs real (0).
    Adds minor noise to make the task non-trivial.
    """
    half = n_samples // 2
    rows = []

    for _ in range(half):
        text = _fill(random.choice(REAL_TEMPLATES))
        rows.append({"text": text, "label": 0})

    for _ in range(half):
        text = _fill(random.choice(FAKE_TEMPLATES))
        rows.append({"text": text, "label": 1})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[data]  Dataset written → {output_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_dataset()
