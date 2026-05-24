"""
evaluate_metrics.py
-------------------
Évaluation des résumés générés :
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore (précision, rappel, F1)
  - Longueur moyenne des résumés
  - Comparaison graphique des modèles
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ROUGE
def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calcule ROUGE-1, ROUGE-2, ROUGE-L (scores F1 moyens)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "ROUGE-1": round(np.mean(r1), 4),
        "ROUGE-2": round(np.mean(r2), 4),
        "ROUGE-L": round(np.mean(rl), 4),
    }


# BERTScore
def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    batch_size: int = 8,
) -> Dict[str, float]:
    """Calcule BERTScore P/R/F1."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang=lang, batch_size=batch_size, verbose=False)
    return {
        "BERTScore-P":  round(P.mean().item(), 4),
        "BERTScore-R":  round(R.mean().item(), 4),
        "BERTScore-F1": round(F1.mean().item(), 4),
    }


# Longueur moyenne
def compute_length_stats(predictions: List[str]) -> Dict[str, float]:
    lengths = [len(p.split()) for p in predictions]
    return {
        "avg_length": round(np.mean(lengths), 2),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }


# Évaluation complète d'un modèle
def evaluate_model(
    model_name: str,
    predictions: List[str],
    references: List[str],
    use_bertscore: bool = True,
    lang: str = "en",
) -> Dict[str, float]:
    """Agrège toutes les métriques pour un modèle donné."""
    logger.info("Évaluation de : %s (%d exemples)", model_name, len(predictions))
    metrics = {"model": model_name}
    metrics.update(compute_rouge(predictions, references))
    if use_bertscore:
        metrics.update(compute_bertscore(predictions, references, lang=lang))
    metrics.update(compute_length_stats(predictions))
    return metrics


# Comparaison multi-modèles
def compare_models(results: List[Dict[str, float]], save: bool = True) -> pd.DataFrame:
    """
    Prend une liste de dicts (un par modèle) et produit :
      - un DataFrame récapitulatif
      - un graphique comparatif
    """
    df = pd.DataFrame(results).set_index("model")

    # Sauvegarde CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path)
    logger.info("Résultats sauvegardés : %s", csv_path)

    # Graphique ROUGE
    rouge_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    rouge_df = df[rouge_cols]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison des modèles — Résumé automatique d'articles scientifiques",
                 fontsize=13, fontweight="bold")

    # Barplot ROUGE
    rouge_df.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="white", width=0.7)
    axes[0].set_title("Scores ROUGE (F1)")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    # BERTScore F1 si disponible
    if "BERTScore-F1" in df.columns:
        df["BERTScore-F1"].plot(kind="bar", ax=axes[1], color="#4C72B0", edgecolor="white", width=0.5)
        axes[1].set_title("BERTScore F1")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure sauvegardée : %s", path)
    plt.show()

    print("\nTableau récapitulatif")
    print(df.to_string())
    return df


# Exemples qualitatifs
def show_examples(
    articles: List[str],
    references: List[str],
    model_summaries: Dict[str, List[str]],
    n: int = 3,
    save: bool = True,
) -> None:
    """Affiche n exemples côte à côte et sauvegarde en CSV."""
    rows = []
    for i in range(min(n, len(articles))):
        row = {
            "article_excerpt": articles[i][:300] + "...",
            "reference":       references[i],
        }
        for model_name, summaries in model_summaries.items():
            row[model_name] = summaries[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        path = os.path.join(RESULTS_DIR, "qualitative_examples.csv")
        df.to_csv(path, index=False)
        logger.info("Exemples qualitatifs sauvegardés : %s", path)

    for i, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"ARTICLE (extrait) :\n{row['article_excerpt']}")
        print(f"\nRÉFÉRENCE :\n{row['reference']}")
        for model_name in model_summaries:
            print(f"\n{model_name.upper()} :\n{row[model_name]}")
    print("=" * 70)


if __name__ == "__main__":
    # Test rapide avec des données factices
    preds = ["The model shows improved performance on summarization tasks."] * 5
    refs  = ["We present a new approach to scientific text summarization using LLMs."] * 5

    r = compute_rouge(preds, refs)
    print("ROUGE:", r)

    l = compute_length_stats(preds)
    print("Longueurs:", l)
