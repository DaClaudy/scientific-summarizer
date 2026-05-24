"""
main.py
-------
Pipeline principal :
  1. Chargement des données (arXiv + PubMed)
  2. EDA
  3. Inférence zero-shot (BART, T5)
  4. Fine-tuning T5-small
  5. Évaluation et comparaison des modèles
  6. Exemples qualitatifs

Usage :
  python main.py --mode all          # pipeline complet
  python main.py --mode zero_shot    # zero-shot uniquement
  python main.py --mode finetune     # fine-tuning uniquement
  python main.py --mode evaluate     # évaluation uniquement (modèle déjà entraîné)
"""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Arguments CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Résumé automatique d'articles scientifiques")
    parser.add_argument("--mode",         type=str, default="all",
                        choices=["all", "zero_shot", "finetune", "evaluate", "eda"],
                        help="Mode d'exécution")
    parser.add_argument("--n_train",      type=int, default=2000,  help="Exemples d'entraînement")
    parser.add_argument("--n_val",        type=int, default=200,   help="Exemples de validation")
    parser.add_argument("--n_test",       type=int, default=200,   help="Exemples de test")
    parser.add_argument("--epochs",       type=int, default=3,     help="Epochs de fine-tuning")
    parser.add_argument("--batch_size",   type=int, default=4,     help="Taille des batches")
    parser.add_argument("--base_model",   type=str, default="t5-small",
                        help="Modèle de base pour fine-tuning")
    parser.add_argument("--source",       type=str, default="both",
                        choices=["arxiv", "pubmed", "both"],
                        help="Source des données")
    parser.add_argument("--no_bertscore", action="store_true",
                        help="Désactive le calcul BERTScore (plus rapide)")
    parser.add_argument("--device",       type=int, default=-1,
                        help="Device GPU (-1 = CPU, 0 = GPU 0)")
    parser.add_argument("--n_examples",   type=int, default=3,
                        help="Nombre d'exemples qualitatifs à afficher")
    return parser.parse_args()


# Pipeline
def main():
    args = parse_args()
    logger.info("Mode : %s | Source : %s | Modèle : %s", args.mode, args.source, args.base_model)

    # Import local (permet d'exécuter depuis la racine du projet)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.data_loader import load_combined
    from src.eda import run_eda
    from src.summarizer import ZeroShotSummarizer, FineTunedSummarizer, lead3_baseline
    from src.evaluate_metrics import evaluate_model, compare_models, show_examples

    # 1. Chargement des données
    logger.info("Chargement des données...")
    dataset = load_combined(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        source=args.source,
    )

    test_articles  = dataset["test"]["article"]
    test_summaries = dataset["test"]["summary"]

    # 2. EDA
    if args.mode in ("all", "eda"):
        logger.info("ÉTAPE 1 : Analyse exploratoire")
        run_eda(dataset)

    # 3. Baseline Lead-3
    results = []
    model_summaries = {}

    logger.info("Baseline Lead-3")
    lead3_preds = lead3_baseline(test_articles)
    model_summaries["Lead-3"] = lead3_preds
    results.append(evaluate_model(
        "Lead-3", lead3_preds, test_summaries,
        use_bertscore=not args.no_bertscore,
    ))

    # 4. Zero-shot
    if args.mode in ("all", "zero_shot"):
        logger.info("ÉTAPE 2 : Inférence zero-shot")

        for model_key in ["bart", "t5"]:
            try:
                zs = ZeroShotSummarizer(model_name=model_key, device=args.device)
                preds = zs.summarize(test_articles, batch_size=args.batch_size)
                model_summaries[f"ZeroShot-{model_key.upper()}"] = preds
                results.append(evaluate_model(
                    f"ZeroShot-{model_key.upper()}", preds, test_summaries,
                    use_bertscore=not args.no_bertscore,
                ))
            except Exception as e:
                logger.warning("Erreur modèle %s : %s", model_key, e)

    # 5. Fine-tuning
    if args.mode in ("all", "finetune"):
        logger.info("ÉTAPE 3 : Fine-tuning %s", args.base_model)
        ft = FineTunedSummarizer(base_model=args.base_model)
        ft.train(
            dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        ft_preds = ft.summarize(test_articles)
        model_summaries[f"FineTuned-{args.base_model}"] = ft_preds
        results.append(evaluate_model(
            f"FineTuned-{args.base_model}", ft_preds, test_summaries,
            use_bertscore=not args.no_bertscore,
        ))

    # 6. Chargement fine-tuné seul
    if args.mode == "evaluate":
        logger.info("Chargement modèle fine-tuné %s pour évaluation", args.base_model)
        ft = FineTunedSummarizer(base_model=args.base_model)
        ft.load_finetuned()
        ft_preds = ft.summarize(test_articles)
        model_summaries[f"FineTuned-{args.base_model}"] = ft_preds
        results.append(evaluate_model(
            f"FineTuned-{args.base_model}", ft_preds, test_summaries,
            use_bertscore=not args.no_bertscore,
        ))

    # 7. Comparaison
    if results:
        logger.info("ÉTAPE 4 : Comparaison des modèles")
        compare_models(results)

        logger.info("ÉTAPE 5 : Exemples qualitatifs")
        show_examples(
            test_articles,
            test_summaries,
            model_summaries,
            n=args.n_examples,
        )

    logger.info("Pipeline terminé")


if __name__ == "__main__":
    main()
