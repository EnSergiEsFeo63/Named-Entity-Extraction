# Named Entity Recognition with CRF (CoNLL-2002 & CADEC)

A Named Entity Recognition (NER) system built using **Conditional Random Fields (CRF)**, trained and evaluated on the **CoNLL-2002** corpus (Spanish and Dutch) and the **CADEC** (Consumer Adverse Drug Event Corpus). The system supports multiple tagging schemes, a configurable feature set, and systematic ablation analysis.

**Authors:** Sergi & Sam

---

## Project Overview

This notebook implements a full NER pipeline:

1. Data loading and exploration (CoNLL-2002: Spanish and Dutch)
2. Feature engineering with a configurable `CRFFeatureGenerator`
3. Model training via NLTK's `CRFTagger`
4. Evaluation using token-level balanced accuracy and entity-level precision/recall/F1
5. Systematic **ablation study** to identify the best feature combination
6. **Tagging scheme comparison**: IO, BIO, BIOE, BIOS, BIOES
7. **Cross-language comparison**: Spanish vs. Dutch
8. Extension to the **CADEC** biomedical corpus

Entity types covered: `LOC`, `MISC`, `ORG`, `PER` (CoNLL-2002) and `ADR`, `Di`, `Dr`, `S`, `F` (CADEC).

---

## Repository Structure

```
.
├── Prac3_Sergi_Sam.ipynb   # Main notebook
├── locations.txt            # Gazetteer: world locations (Spanish & Dutch)
├── person_names.txt         # Gazetteer: person names (Spanish & Dutch)
├── cadec/
│   ├── train.conll          # CADEC training set (CoNLL format)
│   └── test.conll           # CADEC test set (CoNLL format)
└── README.md
```

---

## Requirements

### Python version
Python 3.8 or higher is recommended.

### Dependencies

Install all required packages with:

```bash
pip install nltk scikit-learn pandas seaborn matplotlib python-crfsuite
```

| Package | Purpose |
|---|---|
| `nltk` | CoNLL-2002 corpus, CRFTagger, POS tagging, WordNet lemmatizer |
| `scikit-learn` | Evaluation metrics (`classification_report`, `balanced_accuracy_score`) |
| `pandas` | Results display and analysis |
| `seaborn` + `matplotlib` | Correlation heatmaps and visualizations |
| `python-crfsuite` | Underlying CRF implementation used by NLTK's `CRFTagger` |

### NLTK data downloads

The notebook downloads the following NLTK resources automatically on first run:

```python
nltk.download('conll2002')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

---

## Required Files

Two external files must be placed in the **root of the repository** (same directory as the notebook):

### `locations.txt`
A plain-text file with one location name per line (cities, countries, regions) in Spanish and Dutch. The gazetteer used during execution contains **106,150 unique locations**.

### `person_names.txt`
A plain-text file with one person name per line in Spanish and Dutch. The gazetteer used during execution contains **123,466 unique names**.

### CADEC corpus (`cadec/` folder)
The CADEC corpus is **not publicly distributed via NLTK** and must be obtained separately from the [CSIRO Data Access Portal](https://data.csiro.au/). Once downloaded, place the files as follows:

```
cadec/
├── train.conll
└── test.conll
```

The files are expected to be in CoNLL format (4 columns: token, POS, chunk, NER tag).

> **Note:** The CoNLL-2002 corpus (`esp.*` and `ned.*` files) is downloaded automatically by NLTK. No manual setup is needed for it.

---

## How to Run

1. Clone the repository and navigate to the project folder.
2. Install dependencies (see above).
3. Place `locations.txt` and `person_names.txt` in the root directory.
4. (Optional) Place the CADEC corpus under `cadec/` for the final section.
5. Open and run the notebook:

```bash
jupyter notebook Prac3_Sergi_Sam.ipynb
```

Run all cells in order. Each section builds on the previous one.

---

## Dataset Statistics

| Language | Train sentences | Dev sentences | Test sentences |
|---|---|---|---|
| Spanish | 8,323 | 1,915 | 1,517 |
| Dutch | 15,806 | 2,895 | 5,195 |

Tag distribution in the Spanish training set:

| Tag | Count |
|---|---|
| O | 231,920 |
| B-ORG | 7,390 |
| I-ORG | 4,992 |
| B-LOC | 4,913 |
| B-PER | 4,321 |
| I-PER | 3,903 |
| I-MISC | 3,212 |
| B-MISC | 2,173 |
| I-LOC | 1,891 |

---

## Key Results

### Best feature configuration

Found via ablation study over 8 feature groups. The optimal configuration (maximising Balanced Accuracy on the Spanish dev set) is:

| Feature | Enabled |
|---|---|
| `word_form` | ✅ |
| `morphology` | ✅ |
| `gazetteers` | ✅ |
| `context` | ✅ |
| `pos` | ❌ |
| `prefix_suffix` | ❌ |
| `length` | ❌ |
| `position` | ❌ |

This configuration is labelled **"Base + context"** in the ablation table, where "Base" already includes `word_form`, `morphology`, and `gazetteers`.

### Ablation study — top configurations (Spanish dev, sample 2,500 sentences)

| Configuration | Balanced Acc. | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Completa - length | 0.742 | 0.702 | 0.719 | 0.686 |
| Completa - prefix_suffix | 0.735 | 0.692 | 0.711 | 0.674 |
| Completa - position | 0.729 | 0.705 | 0.720 | 0.691 |
| Completa - context | 0.728 | 0.714 | 0.731 | 0.698 |
| Completa | 0.722 | 0.695 | 0.710 | 0.681 |
| Base + context | 0.722 | 0.677 | 0.693 | 0.662 |

> Removing individual features from the full configuration often outperforms adding them, suggesting negative feature interactions — especially from `prefix_suffix`, `length`, and `position`.

### Tagging scheme comparison (Spanish dev, best config)

| Scheme | Balanced Acc. | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **BIO** | **0.682** | **0.628** | 0.651 | 0.606 |
| IO | 0.677 | 0.619 | 0.647 | 0.594 |
| BIOE | 0.673 | 0.633 | 0.653 | 0.613 |
| BIOS | 0.665 | 0.621 | 0.645 | 0.598 |
| BIOES | 0.664 | 0.625 | 0.644 | 0.607 |

BIO was selected as the default scheme given its slightly superior and more stable metrics across both languages.

### Final model — dev and test results (full training data)

#### Dev set

| Language | Balanced Acc. | F1 Score | Precision | Recall | LOC | MISC | ORG | PER |
|---|---|---|---|---|---|---|---|---|
| Spanish | 0.761 | 0.713 | 0.731 | 0.697 | 0.785 | 0.356 | 0.689 | 0.762 |
| Dutch | 0.749 | 0.689 | 0.737 | 0.647 | 0.752 | 0.593 | 0.522 | 0.755 |

#### Test set

| Language | Balanced Acc. | F1 Score | Precision | Recall | LOC | MISC | ORG | PER |
|---|---|---|---|---|---|---|---|---|
| Spanish | **0.800** | **0.775** | 0.787 | 0.763 | 0.761 | 0.415 | 0.791 | **0.873** |
| Dutch | **0.759** | **0.711** | 0.753 | 0.673 | 0.748 | 0.557 | 0.592 | **0.811** |

The model performs best on `PER` entities in both languages, and struggles most with `MISC`, which is expected given its heterogeneous nature.

---

## Notes

- CRF model files (`.crf`) are generated during training and saved to the working directory. These are not versioned and can be safely deleted between runs.
- The CADEC section uses a dummy POS tag (`"NN"`) since CADEC does not include POS annotations in this format.
- All evaluation is internally normalised to IO format for cross-scheme consistency.
- Training with the full CoNLL-2002 datasets takes several minutes. Reduce `sample_size` in `run_experiment()` for faster iteration during development.
