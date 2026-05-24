"""
eda.py
------
Analyse exploratoire des données (EDA) :
distribution des longueurs, nuages de mots, statistiques descriptives.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# Statistiques descriptives
def compute_stats(dataset, split: str = "train") -> pd.DataFrame:
    """Calcule longueur (mots) des articles et résumés."""
    records = []
    for sample in dataset[split]:
        art_len = len(sample["article"].split())
        sum_len = len(sample["summary"].split())
        records.append({
            "source": sample.get("source", "unknown"),
            "article_words": art_len,
            "summary_words": sum_len,
            "compression_ratio": round(sum_len / art_len, 4) if art_len > 0 else 0,
        })
    df = pd.DataFrame(records)
    logger.info("\nStatistiques descriptives [%s]\n%s", split, df.describe().round(2))
    return df


# Visualisations
def plot_length_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """Distribution des longueurs d'articles et de résumés par source."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Distribution des longueurs — Articles scientifiques", fontsize=14, fontweight="bold")

    palette = {"arxiv": "#4C72B0", "pubmed": "#DD8452", "unknown": "#55A868"}

    # Articles
    sns.histplot(data=df, x="article_words", hue="source", bins=50,
                 palette=palette, ax=axes[0], kde=True, alpha=0.6)
    axes[0].set_title("Longueur des articles (mots)")
    axes[0].set_xlabel("Nombre de mots")

    # Résumés
    sns.histplot(data=df, x="summary_words", hue="source", bins=50,
                 palette=palette, ax=axes[1], kde=True, alpha=0.6)
    axes[1].set_title("Longueur des résumés (mots)")
    axes[1].set_xlabel("Nombre de mots")

    # Ratio de compression
    sns.histplot(data=df, x="compression_ratio", hue="source", bins=40,
                 palette=palette, ax=axes[2], kde=True, alpha=0.6)
    axes[2].set_title("Ratio de compression (résumé / article)")
    axes[2].set_xlabel("Ratio")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "length_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure sauvegardée : %s", path)
    plt.show()


def plot_source_breakdown(df: pd.DataFrame, save: bool = True) -> None:
    """Répartition par source."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["source"].value_counts()
    colors = ["#4C72B0", "#DD8452"]
    counts.plot.pie(autopct="%1.1f%%", colors=colors, ax=ax, startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Répartition du corpus par source", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "source_breakdown.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure sauvegardée : %s", path)
    plt.show()


def plot_top_words(dataset, split: str = "train", n: int = 30, save: bool = True) -> None:
    """Top N mots les plus fréquents dans les résumés."""
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    stop_en = set(stopwords.words("english"))
    stop_fr = set(stopwords.words("french"))
    stop = stop_en | stop_fr | {"also", "using", "used", "based", "results", "show",
                                 "shown", "two", "one", "may", "however", "et", "al"}

    words = []
    for sample in dataset[split]:
        tokens = sample["summary"].lower().split()
        words.extend([w for w in tokens if w.isalpha() and w not in stop and len(w) > 3])

    counter = Counter(words)
    top = counter.most_common(n)
    labels, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(labels), y=list(counts), palette="Blues_r", ax=ax)
    ax.set_title(f"Top {n} mots les plus fréquents dans les résumés", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mot")
    ax.set_ylabel("Fréquence")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "top_words.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure sauvegardée : %s", path)
    plt.show()


def run_eda(dataset) -> pd.DataFrame:
    """Lance l'EDA complète sur le dataset."""
    logger.info("Démarrage de l'EDA...")
    df = compute_stats(dataset, split="train")
    plot_length_distributions(df)
    plot_source_breakdown(df)
    plot_top_words(dataset, split="train")
    return df


if __name__ == "__main__":
    from data_loader import load_combined
    ds = load_combined(n_train=500, n_val=100, n_test=100, source="both")
    run_eda(ds)
