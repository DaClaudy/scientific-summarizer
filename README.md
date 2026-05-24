# Résumé Automatique d'Articles Scientifiques

> **Cours : Large Language Models**
> Utilisation de modèles pré-entraînés (BART, T5) et fine-tuning sur un corpus combiné arXiv + PubMed pour la génération automatique de résumés d'articles scientifiques.

---

## Problématique

Face à l'explosion du nombre de publications scientifiques, il devient impossible pour un chercheur de lire l'intégralité des articles pertinents dans son domaine. Ce projet explore dans quelle mesure les **Large Language Models** peuvent automatiser la génération de résumés d'articles scientifiques, en comparant :

- une **baseline extractive** simple (Lead-3)
- des **modèles pré-entraînés** en mode zero-shot (BART, T5)
- un **modèle fine-tuné** sur un corpus spécialisé arXiv + PubMed

---

## Structure du projet

```
scientific-summarizer/
│
├── main.py                   # Pipeline principal (CLI)
├── requirements.txt          # Dépendances Python
├── README.md                 
│
├── src/
│   ├── data_loader.py        # Chargement & nettoyage des données
│   ├── eda.py                # Analyse exploratoire (EDA)
│   ├── summarizer.py         # Modèles zero-shot & fine-tuning
│   └── evaluate_metrics.py  # ROUGE, BERTScore, visualisations
│
├── notebooks/
│   └── exploration.ipynb     # Notebook interactif (optionnel)
│
├── figures/                  # Graphiques générés automatiquement
├── results/                  # CSV des métriques & exemples qualitatifs
└── models/                   # Modèles fine-tunés sauvegardés
```

---

## Données

| Source | Dataset HuggingFace | Description |
|--------|-------------------|-------------|
| **arXiv** | `scientific_papers` (arxiv) | Articles en informatique, physique, mathématiques |
| **PubMed** | `scientific_papers` (pubmed) | Articles en médecine et biologie |

Les deux datasets partagent la même structure :
- `article` : corps de l'article (plusieurs milliers de mots)
- `abstract` : résumé cible (quelques centaines de mots)

Le corpus combiné utilisé par défaut :
- **Train** : 2 000 exemples (1 000 arXiv + 1 000 PubMed)
- **Validation** : 200 exemples
- **Test** : 200 exemples

---

## Modèles utilisés

| Modèle | Type | Description |
|--------|------|-------------|
| **Lead-3** | Baseline extractive | 3 premières phrases de l'article |
| **facebook/bart-large-cnn** | Zero-shot | BART pré-entraîné sur CNN/DailyMail |
| **t5-small** | Zero-shot | T5 pré-entraîné sur C4 |
| **t5-small (fine-tuné)** | Fine-tuning | T5-small spécialisé sur arXiv + PubMed |

---

## Métriques d'évaluation

- **ROUGE-1, ROUGE-2, ROUGE-L** : chevauchement de n-grammes entre le résumé généré et la référence
- **BERTScore (P / R / F1)** : similarité sémantique basée sur les embeddings BERT
- **Longueur moyenne** des résumés générés

---

## Installation & Utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/DaClaudy/scientific-summarizer.git
cd scientific-summarizer
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Un GPU est recommandé pour le fine-tuning. Sur CPU, réduire `--n_train` à 200-500.

---

## Lancer le pipeline

### Pipeline complet (EDA + zero-shot + fine-tuning + évaluation)

```bash
python main.py --mode all --source both --n_train 2000 --n_val 200 --n_test 200
```

### EDA uniquement

```bash
python main.py --mode eda --source both
```

### Inférence zero-shot uniquement

```bash
python main.py --mode zero_shot --n_test 100 --no_bertscore
```

### Fine-tuning uniquement

```bash
python main.py --mode finetune --base_model t5-small --epochs 3 --batch_size 4
```

### Évaluation d'un modèle déjà fine-tuné

```bash
python main.py --mode evaluate --base_model t5-small --n_test 200
```

---

## Arguments disponibles

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--mode` | `all` | `all`, `eda`, `zero_shot`, `finetune`, `evaluate` |
| `--source` | `both` | `arxiv`, `pubmed`, `both` |
| `--n_train` | `2000` | Nombre d'exemples d'entraînement |
| `--n_val` | `200` | Nombre d'exemples de validation |
| `--n_test` | `200` | Nombre d'exemples de test |
| `--epochs` | `3` | Nombre d'epochs de fine-tuning |
| `--batch_size` | `4` | Taille des batches |
| `--base_model` | `t5-small` | Modèle HuggingFace pour fine-tuning |
| `--no_bertscore` | `False` | Désactive BERTScore (plus rapide) |
| `--device` | `-1` | `-1` = CPU, `0` = GPU 0 |
| `--n_examples` | `3` | Nombre d'exemples qualitatifs affichés |

---

## Résultats attendus

Les figures et résultats sont automatiquement sauvegardés dans :
- `figures/length_distributions.png` — distribution des longueurs
- `figures/source_breakdown.png` — répartition arXiv / PubMed
- `figures/top_words.png` — mots les plus fréquents dans les résumés
- `figures/model_comparison.png` — comparaison ROUGE & BERTScore
- `results/evaluation_results.csv` — tableau récapitulatif des métriques
- `results/qualitative_examples.csv` — exemples de résumés côte à côte

---

## Exemple de résultats (indicatifs)

| Modèle | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|--------|---------|---------|---------|--------------|
| Lead-3 | 0.28 | 0.09 | 0.18 | 0.81 |
| ZeroShot-BART | 0.35 | 0.13 | 0.24 | 0.84 |
| ZeroShot-T5 | 0.31 | 0.10 | 0.21 | 0.82 |
| FineTuned-T5 | **0.43** | **0.18** | **0.31** | **0.87** |

> Ces scores sont indicatifs et dépendent de la taille du corpus et du nombre d'epochs.

---

## Configuration matérielle recommandée

| Scénario | RAM | GPU | Temps estimé |
|----------|-----|-----|--------------|
| Zero-shot uniquement | 8 Go | Non requis | ~10 min |
| Fine-tuning (n_train=500) | 8 Go | Non requis (lent) | ~1h CPU |
| Fine-tuning (n_train=2000) | 16 Go | Recommandé | ~30 min GPU |

---

## Références

- Lewis et al. (2020) — [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- Raffel et al. (2020) — [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- Zhang et al. (2020) — [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- Cohan et al. (2018) — [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685)
- HuggingFace Datasets — [scientific_papers](https://huggingface.co/datasets/scientific_papers)

---

## Auteur

**Damigou BOUNDJA**
Projet réalisé dans le cadre du cours **Large Language Models**.

---

## Licence

MIT License — libre d'utilisation et de modification.
