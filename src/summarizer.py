"""
summarizer.py
-------------
Deux modes :
  1. Pipeline HuggingFace (zero-shot) : facebook/bart-large-cnn, t5-small
  2. Fine-tuning sur sous-ensemble arXiv + PubMed avec Seq2SeqTrainer
"""

import os
import logging
import torch
from typing import List, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline,
    set_seed,
)
from datasets import DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_seed(42)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Modèles disponibles
PRETRAINED_MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5":   "t5-small",
    "pegasus": "google/pegasus-xsum",
}


# 1. Pipeline zero-shot
class ZeroShotSummarizer:
    """Résumé avec modèle pré-entraîné, sans fine-tuning."""

    def __init__(self, model_name: str = "bart", device: int = -1):
        model_id = PRETRAINED_MODELS.get(model_name, model_name)
        logger.info("Chargement du modèle zero-shot : %s", model_id)
        self.pipe = pipeline(
            "summarization",
            model=model_id,
            device=device,
            truncation=True,
        )
        self.model_name = model_name

    def summarize(
        self,
        texts: List[str],
        max_length: int = 256,
        min_length: int = 40,
        batch_size: int = 4,
    ) -> List[str]:
        results = self.pipe(
            texts,
            max_length=max_length,
            min_length=min_length,
            batch_size=batch_size,
            do_sample=False,
        )
        return [r["summary_text"] for r in results]


# 2. Fine-tuning avec Seq2SeqTrainer
class FineTunedSummarizer:
    """Fine-tuning d'un modèle seq2seq sur arXiv + PubMed."""

    def __init__(self, base_model: str = "t5-small", output_dir: Optional[str] = None):
        self.base_model = base_model
        self.output_dir = output_dir or os.path.join(MODELS_DIR, f"finetuned_{base_model.replace('/', '_')}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        logger.info("Modèle de base chargé : %s", base_model)

    def _tokenize(self, dataset: DatasetDict, prefix: str = "summarize: ") -> DatasetDict:
        """Tokenise le dataset pour seq2seq."""
        from src.data_loader import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH

        def _tok(batch):
            inputs = [prefix + doc for doc in batch["article"]]
            model_inputs = self.tokenizer(
                inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length"
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        cols_to_remove = [c for c in dataset["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
        return dataset.map(_tok, batched=True, remove_columns=cols_to_remove)

    def train(
        self,
        dataset: DatasetDict,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        fp16: bool = False,
    ):
        tokenized = self._tokenize(dataset)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, pad_to_multiple_of=8)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=fp16 and torch.cuda.is_available(),
            report_to="none",
            learning_rate=learning_rate,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        logger.info("Démarrage du fine-tuning sur %d exemples...", len(tokenized["train"]))
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Modèle fine-tuné sauvegardé dans : %s", self.output_dir)
        return trainer

    def load_finetuned(self, path: Optional[str] = None):
        """Charge un modèle fine-tuné depuis le disque."""
        load_path = path or self.output_dir
        logger.info("Chargement du modèle fine-tuné depuis : %s", load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)

    def summarize(
        self,
        texts: List[str],
        max_length: int = 256,
        min_length: int = 40,
        prefix: str = "summarize: ",
    ) -> List[str]:
        summaries = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for text in texts:
            inputs = self.tokenizer(
                prefix + text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            summaries.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        return summaries


# Baseline extractive (Lead-3)
def lead3_baseline(texts: List[str]) -> List[str]:
    """Baseline simple : retourne les 3 premières phrases de l'article."""
    import re
    summaries = []
    for text in texts:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summaries.append(" ".join(sentences[:3]))
    return summaries


if __name__ == "__main__":
    sample = (
        "Large language models (LLMs) have demonstrated remarkable capabilities across "
        "a wide range of natural language processing tasks. In this paper, we investigate "
        "the use of transformer-based architectures for automatic summarization of scientific "
        "articles. We fine-tune a T5-small model on a combined corpus of arXiv and PubMed "
        "articles and evaluate performance using ROUGE metrics. Our results show significant "
        "improvements over the Lead-3 baseline, with ROUGE-1 scores improving by 8 points."
    )
    zs = ZeroShotSummarizer(model_name="t5")
    print("Zero-shot:", zs.summarize([sample])[0])
    print("Lead-3   :", lead3_baseline([sample])[0])
