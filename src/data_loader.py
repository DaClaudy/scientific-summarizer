"""
data_loader.py
--------------
Chargement et prétraitement des datasets arXiv et PubMed
pour la tâche de résumé automatique d'articles scientifiques.
"""

import re
import logging
from typing import Optional
from datasets import load_dataset, DatasetDict, concatenate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constantes
MAX_INPUT_LENGTH = 1024   # tokens max pour l'article (tronqué)
MAX_TARGET_LENGTH = 256   # tokens max pour le résumé


# Nettoyage textuel
def clean_text(text: str) -> str:
    """Nettoyage basique : espaces multiples, caractères spéciaux, LaTeX résiduel."""
    if not isinstance(text, str):
        return ""
    # Supprime les commandes LaTeX simples
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " [FORMULA] ", text)
    # Espaces / retours à la ligne multiples
    text = re.sub(r"\s+", " ", text)
    # Caractères de contrôle
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def preprocess_sample(sample: dict, article_col: str, summary_col: str) -> dict:
    """Applique clean_text sur article et résumé."""
    return {
        "article": clean_text(sample[article_col]),
        "summary": clean_text(sample[summary_col]),
    }


def filter_sample(sample: dict, min_article: int = 200, min_summary: int = 30) -> bool:
    """Filtre les exemples trop courts ou vides."""
    return (
        len(sample["article"]) >= min_article
        and len(sample["summary"]) >= min_summary
    )


# Chargement arXiv
def load_arxiv(n_train: int = 5000, n_val: int = 500, n_test: int = 500) -> DatasetDict:
    """
    Charge le dataset scientific_papers / arXiv depuis HuggingFace.
    Colonnes : 'article' (corps), 'abstract' (résumé cible).
    """
    logger.info("Chargement du dataset arXiv...")
    ds = load_dataset("scientific_papers", "arxiv", trust_remote_code=True)

    def _prep(sample):
        return preprocess_sample(sample, "article", "abstract")

    ds = ds.map(_prep, remove_columns=ds["train"].column_names)
    ds = ds.filter(filter_sample)

    return DatasetDict({
        "train": ds["train"].select(range(min(n_train, len(ds["train"])))),
        "validation": ds["validation"].select(range(min(n_val, len(ds["validation"])))),
        "test": ds["test"].select(range(min(n_test, len(ds["test"])))),
    })


# Chargement PubMed
def load_pubmed(n_train: int = 5000, n_val: int = 500, n_test: int = 500) -> DatasetDict:
    """
    Charge le dataset scientific_papers / PubMed depuis HuggingFace.
    Colonnes identiques : 'article', 'abstract'.
    """
    logger.info("Chargement du dataset PubMed...")
    ds = load_dataset("scientific_papers", "pubmed", trust_remote_code=True)

    def _prep(sample):
        return preprocess_sample(sample, "article", "abstract")

    ds = ds.map(_prep, remove_columns=ds["train"].column_names)
    ds = ds.filter(filter_sample)

    return DatasetDict({
        "train": ds["train"].select(range(min(n_train, len(ds["train"])))),
        "validation": ds["validation"].select(range(min(n_val, len(ds["validation"])))),
        "test": ds["test"].select(range(min(n_test, len(ds["test"])))),
    })


# Fusion arXiv + PubMed
def load_combined(
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 500,
    source: str = "both",          # "arxiv" | "pubmed" | "both"
) -> DatasetDict:
    """
    Retourne un DatasetDict combiné (ou filtré par source).
    Ajoute une colonne 'source' pour traçabilité.
    """
    if source == "arxiv":
        ds = load_arxiv(n_train, n_val, n_test)
        for split in ds:
            ds[split] = ds[split].add_column("source", ["arxiv"] * len(ds[split]))
        return ds

    if source == "pubmed":
        ds = load_pubmed(n_train, n_val, n_test)
        for split in ds:
            ds[split] = ds[split].add_column("source", ["pubmed"] * len(ds[split]))
        return ds

    # both
    half_train = n_train // 2
    half_val   = n_val   // 2
    half_test  = n_test  // 2

    arxiv  = load_arxiv(half_train, half_val, half_test)
    pubmed = load_pubmed(half_train, half_val, half_test)

    for split in arxiv:
        arxiv[split]  = arxiv[split].add_column("source",  ["arxiv"]  * len(arxiv[split]))
        pubmed[split] = pubmed[split].add_column("source", ["pubmed"] * len(pubmed[split]))

    combined = DatasetDict({
        split: concatenate_datasets([arxiv[split], pubmed[split]]).shuffle(seed=42)
        for split in arxiv
    })

    logger.info(
        "Dataset combiné — train: %d | val: %d | test: %d",
        len(combined["train"]), len(combined["validation"]), len(combined["test"]),
    )
    return combined


# Tokenisation pour seq2seq
def tokenize_dataset(dataset: DatasetDict, tokenizer, prefix: str = "summarize: ") -> DatasetDict:
    """
    Tokenise article → input_ids / attention_mask
    et summary → labels, pour entraînement seq2seq (T5/BART).
    """
    def _tokenize(batch):
        inputs = [prefix + doc for doc in batch["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_tokenize, batched=True, remove_columns=["article", "summary", "source"])


# Point d'entrée rapide
if __name__ == "__main__":
    ds = load_combined(n_train=100, n_val=20, n_test=20, source="both")
    print(ds)
    print("\nExemple d'article (extrait) :")
    print(ds["train"][0]["article"][:300])
    print("\nRésumé correspondant :")
    print(ds["train"][0]["summary"])
    print("\nSource :", ds["train"][0]["source"])
