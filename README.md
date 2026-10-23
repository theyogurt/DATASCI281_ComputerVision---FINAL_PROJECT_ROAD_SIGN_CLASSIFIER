# US Road Sign Classification (DATASCI 281 Final Project)

Image classifier that recognizes four road sign types (`speedlimit`, `stop`, `crosswalk`, `trafficlight`) from cropped photos, using the Kaggle [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) dataset (CC0, 877 images, 1,244 labeled signs).

We compare three feature types (simple, hand-designed, pre-trained) across two classifiers and analyze accuracy vs. efficiency.

## Repo layout

```
road-sign-classifier/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # Kaggle download (not committed)
│   ├── processed/    # cropped signs, one folder per class (not committed)
│   └── splits/       # train.csv, val.csv, test.csv (committed)
├── notebooks/
│   ├── 01_data_prep_and_eda.ipynb
│   ├── 02_simple_features_classifiers.ipynb
│   ├── 03_handdesigned_features_classifiers.ipynb
│   ├── 04_pretrained_features_classifiers.ipynb
│   └── 05_results_and_efficiency.ipynb
├── src/
│   └── utils.py      # shared image size, split loading, caching, timing, metrics, result logging
├── features/         # cached .npy feature files (not committed)
├── results/          # results.csv and per_class/ (committed, small)
└── report/           # report PDF and slides
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Download the dataset and unzip it into `data/raw/` (so you have `data/raw/images/` and `data/raw/annotations/`).

## Notebook order and ownership

| Notebook | Purpose | Owner |
| --- | --- | --- |
| 01_data_prep_and_eda | Crop signs, create the frozen split, variation plots |Cesar|
| 02_simple_features_classifiers | HOG + HSV histogram, 2 classifiers |Cesar|
| 03_handdesigned_features_classifiers | Color mask + shape feature, 2 classifiers |Tung La |
| 04_pretrained_features_classifiers | ResNet embeddings (CPU), 2 classifiers |Vansh|
| 05_results_and_efficiency | 6-combo accuracy vs. cost, discussion |Vansh|
| Proposal and repo | Repo setup, proposal |All|
| Report | Assemble PDF from notebook explanations |All|
| Slides | Final presentation, weekly updates |All|

Notebook 01 must produce the split CSVs first. Notebooks 02-04 can prototype on ~20 starter crops per class before that, then run on the full split. Notebook 05 reads `results/results.csv`, which notebooks 02-04 fill in.

## Team rules

- **One frozen split.** Load it with `load_split("train" | "val" | "test")`. Never re-split.
- **Tune on validation only.** Touch the test set once, at the end of each notebook.
- **Same two classifiers** in notebooks 02-04, so the six combos are comparable.
- **Plug-in contract.** A feature function takes an RGB image at `IMAGE_SIZE` and returns a 1-D vector. Cache with `save_features(X, name, split)`; rows follow the split CSV order.
- **Fair timing.** Use `timed(...)` from `utils.py` and run timing on the same machine.
- **Git.** One branch per notebook (for example `nb02-simple-features`), pull request into `main`. Pull `main` before you start each session.

## Group contributions

TODO: fill in what each person worked on before submitting the report.
