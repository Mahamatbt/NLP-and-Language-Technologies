# Vaccine tweet sentiment: sequential model investigation

This project studies the approved **“To Vaccinate or Not to Vaccinate”** tweet sentiment challenge using the files supplied in `MLT1 group`. It follows the task division in `Task_Division.pdf` and the broader requirements in `Formative 2_sep 2026.pdf`.

## Deliverables

- `investigation.ipynb`: reproducible data audit, EDA, two text baselines, optional training/evaluation scaffold for three sequence neural models, and error analysis.
- `report.md`: research framing, methods, local results, interpretation, limitations, references, and explicit status of experiments not runnable in this environment.
- `data/`: copies of source CSVs (the notebook never edits the originals).
- `results/`: generated plots, result tables, and misclassified examples.

## Run

There is one project notebook: `investigation.ipynb`.

### Google Colab

1. Upload/push this project to [Mahamatbt/NLP-and-Language-Technologies](https://github.com/Mahamatbt/NLP-and-Language-Technologies) with `investigation.ipynb`, `data/`, `README.md`, and `report.md`.
2. Open [the notebook in Colab](https://colab.research.google.com/github/Mahamatbt/NLP-and-Language-Technologies/blob/main/investigation.ipynb).
3. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
4. Run the first **Colab setup** cell. It installs dependencies, clones the repository into the runtime so `data/` is present, and checks GPU availability. Then choose **Runtime → Run all**. The XLM-R checkpoint download may take time.
5. Download the completed notebook with **File → Download → Download .ipynb** and commit it back to GitHub if you want to preserve outputs. Colab's temporary `results/` files can be downloaded from the file panel.

The notebook can also run locally with Python 3.10+ and pandas, NumPy, scikit-learn, matplotlib, Jupyter, PyTorch, and Hugging Face Transformers installed. The neural sections need PyTorch; XLM-R also needs Transformers and access to the pretrained checkpoint.

## Target interpretation

The original Zindi challenge evaluates a continuous sentiment score with RMSE. Most labels are -1, 0, or +1, while one valid source row has a fractional target (2/3), retained for regression scoring. For a complementary class view, targets are rounded to the nearest sentiment class and macro-F1 is reported. One blank-label row is removed. One malformed CSV row is repaired using its evident column shift (tweet text in `tweet_id`, hard label in `safe_text`, agreement score in `label`). The notebook records and asserts those rules.

## Reproducibility and caution

All reported baseline numbers use a single stratified 80/20 split (seed 42) over the 10,000 usable training examples. These are validation results, not competition test scores. Hyperparameter tuning was not conducted; neural model performance remains unmeasured until the corresponding cells are run. Avoid presenting the supplied starter notebook's leaderboard scores as this group's experimental results.
