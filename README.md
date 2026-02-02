# Replication Code: BERT, RoBERTa or DeBERTa?

Replication materials for:

> Timoneda, Joan C. and Sebastián Vallejo Vera. "BERT, RoBERTa or DeBERTa?" *Journal of Politics*, 2025.

## Overview

This repository contains code to replicate the analyses comparing transformer-based models (BERT, RoBERTa, DeBERTa) and baseline classifiers (SVM, Logistic Regression, BiLSTM) across three text classification applications in political science.

## Repository Structure

```
.
├── Application 1 - Civility/
│   ├── civility_data.xlsx              # Data
│   ├── Raw Transformers code/          # BERT, RoBERTa, DeBERTa notebooks
│   ├── Baselines code/                 # LSTM, SVM/LR scripts
│   └── Analysis/                       # Results aggregation
│
├── Application 2 - Speeches/
│   ├── speeches_translated.RData       # Data
│   ├── speeches_and_scores_label.csv   # Labels
│   ├── Raw Transformers code/          # mBERT, XLM-RoBERTa, mDeBERTa notebooks
│   ├── Baselines code/                 # LSTM, SVM/LR scripts
│   └── Analysis/                       # Results aggregation
│
├── Application 3 - Fake news/
│   ├── fake_news_covid.xlsx            # Data
│   ├── Raw Transformers code/
│   │   ├── Finetuning/                 # Fine-tuning notebooks
│   │   └── Pretraining/                # Domain-adaptive pretraining
│   ├── Baselines code/                 # LSTM, SVM/LR scripts
│   └── Analysis/                       # Results aggregation
│
└── Supporting material/
    └── Benchmarking/                   # Benchmarking comparisons
```

## Requirements

### Hardware
- GPU with CUDA support recommended for transformer models
- Minimum 16GB GPU memory for DeBERTa (13GB for RoBERTa, less for BERT)

### Software
Install dependencies:
```bash
pip install -r requirements.txt
```

Additionally, download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

For LSTM baselines, you will need GloVe word vectors. Download from https://nlp.stanford.edu/projects/glove/ and place in the appropriate directory.

## Running the Code

### Transformer Models
The transformer fine-tuning code is in Jupyter notebooks (`.ipynb` files) in the `Raw Transformers code/` directories. These can be run in:
- Jupyter Notebook/Lab
- Google Colab (recommended for GPU access)

### Baseline Models
Run the Python scripts from the `Baselines code/` directories:
```bash
python "Application 1 - Civility/Baselines code/SVM_LR_civility.py"
python "Application 1 - Civility/Baselines code/LSTM_civility.py"
```

### Results Aggregation
After running experiments, aggregate results using scripts in `Analysis/` directories.

## Applications

1. **Civility Detection**: Binary classification of civil vs. uncivil online comments
2. **Speech Classification**: Multi-class classification of political speech types (campaign, famous, international, ribbon-cutting)
3. **Fake News Detection**: Three-class classification of COVID-19 related tweets (fake, true, undetermined)

## Notes

- All models use 10-fold cross-validation
- Results are saved with timestamps to track multiple runs
- The `pretraining_tweets_en_full.txt` file for domain-adaptive pretraining is not included due to size constraints

## Citation

```bibtex
@article{timoneda2025bert,
  title={BERT, RoBERTa, or DeBERTa? Comparing Performance Across Transformers Models in Political Science Text},
  author={Timoneda, Joan C and Vera, Sebasti{\'a}n Vallejo},
  journal={The Journal of Politics},
  volume={87},
  number={1},
  pages={347--364},
  year={2025},
  publisher={The University of Chicago Press Chicago, IL}
}
```
