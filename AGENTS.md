# Agent instructions

## Project identity

- This repository is a notebook-based UNSW-NB15 exploratory-analysis and binary-classification project.
- The main binary target is `label` (`0` normal, `1` attack). `attack_cat` is a separate multiclass label and is dropped by the feature-engineering workflow.
- Treat `README.md` as the human-oriented workflow guide. This file is the agent-oriented source of truth for workspace invariants and safe change boundaries.

## Notebook roles and order

1. `notebooks/About Dataset and It's Cleaning.ipynb`: uses global `MAX_RAW_ROWS` (default 50,000) to make a reproducible uniform reservoir sample across four raw traffic CSV parts while streaming; the schema CSV is metadata and the events-list CSV is not traffic data. Creates a random 70/30 train/test split (`random_state=16`), performs selected cleaning, writes the prepared train/test CSVs and metadata to `notebooks/final_ipynb`.
2. `notebooks/Exploratory Data Analysis.ipynb`: reads those prepared CSVs with `keep_default_na=False`; exploratory analysis only. In-memory plot helper columns/values are not persisted to CSV.
3. `notebooks/Feature_Engineering_and_Data_Preparation.ipynb`: reads the prepared CSVs and metadata; performs train-fitted correlation selection, feature engineering, log transforms, scaling, and one-hot encoding; applies the fitted transforms to test data; writes model inputs and preprocessing artifacts to `notebooks/.final_ipynb/`.
4. `notebooks/ML Models and Results.ipynb`: loads the persisted train/test sparse matrices, feature metadata, and preprocessing objects from `./.final_ipynb`, then evaluates models.
5. `notebooks/Prediction of Raw Data.ipynb`: loads the feature-importance random forest and its matching preprocessing artifacts from `./.final_ipynb`; evaluates against the current prepared test CSV and exposes prediction for raw feature rows.

- All four notebooks use relative paths that expect `notebooks/` as the working directory (`../data` for dataset files; `./.final_ipynb` and `./final_ipynb` for artifacts). Keep producer and consumer paths consistent if changing this assumption.
- For any run using a row cap or seed, run the cleaning notebook first to regenerate the prepared CSVs from the fresh raw sample, then run feature engineering and modeling. Do not reuse prepared CSVs from a previous cap/seed. The EDA and feature-engineering notebooks must reject stale prepared CSVs before loading them when their metadata lacks the sample marker or row count.
- `MAX_RAW_ROWS` is user-configurable in the cleaning notebook and must remain at least 10,000 to satisfy the project's minimum valid-record requirement. The default is 50,000 raw records before the train/test split.

## Artifact contract

- Feature-engineering final outputs must be saved after both train and test are fully transformed:
  - `.final_ipynb/final_train.npz` and `.final_ipynb/final_test.npz` = compressed CSR feature matrices.
  - `.final_ipynb/final_train_meta.pkl` and `.final_ipynb/final_test_meta.pkl` = `(feature_columns, labels)`.
- The train/test feature columns must be identical and in the same order; each label vector must match its split's row count.
- Preserve the prepared test split. Never fit scalers/encoders on test data, resplit/concatenate the saved splits, or transform the already-preprocessed test matrix again in the models notebook.
- The model notebook loads CSR matrices directly and retains the saved feature-name list for reporting. Preserve every feature name, order, and numeric value; memory work must change storage only, not the selected inputs or one-hot attributes.
- After splitting `x_train_csr` into train/CV sets, keep each feature matrix paired with its corresponding labels. Any sampled indices for model tuning must be generated from and applied to `x_train_new_csr` and `y_train`, not the original full `x_train_csr`.
- The model notebook saves fitted estimators with pickle under `notebooks/.final_ipynb/` when run from `notebooks/`; keep the preprocessing artifacts and selected feature indices available alongside the models for inference. These local pickles are ignored by Git.
- The model notebook's evaluation helper must use probability or decision scores for ROC-AUC, while F1/confusion matrices use predicted classes. LR, SVM, and decision-tree evaluation must use the selected `best_estimator_`/`best_params_` from their searches. Clear stale saved metric outputs after changing evaluation behavior; do not report old scores as results of corrected code.
- The raw-prediction notebook must use the current `data/test_alldata_EDA.csv` for labeled held-out evaluation, ignore target columns during prediction, and apply the saved train-fitted preprocessing and feature-importance indices exactly. Preserve the 55 selected features in the saved order and keep one-hot/model inputs sparse.
- Keep model estimator serialization as pickle files under `notebooks/.final_ipynb/` using the existing filenames unless explicitly requested otherwise. Rerunning the model notebook should overwrite those artifacts with the newly fitted estimators; retain the same preprocessing objects, selected feature metadata, and feature ordering required by inference.
- Keep one-hot columns sparse when constructing train/test feature tables; never call `.toarray()` on the complete encoded split or `.values` on the complete feature DataFrame to create a sparse matrix.
- Preprocessing artifacts (`saved_dict.pkl`, `mode_dict.pkl`, `scaler.pkl`, and `ohe_*.pkl`) are generated by feature engineering for downstream use. Keep their saved location and model-loader paths aligned.
- `notebooks/final_ipynb` and `notebooks/.final_ipynb/` are distinct paths with different roles; do not merge or rename them without updating both producers and consumers.

## Data integrity and ignore rules

- `data/` contains user/local raw and derived datasets; it is Git-ignored. Never modify or stage dataset files unless explicitly requested.
- `.pkl` files under `notebooks/` are Git-ignored. The metadata pickle `notebooks/final_ipynb` has no `.pkl` extension; do not mistake it for a model-ready dataset.
- Never overwrite the original `train_alldata_EDA.csv` / `test_alldata_EDA.csv` with transformed features. Persist derived artifacts separately.
- The cleaning notebook replaces `train['service'] == '-'` with the string `"None"`; it does not make the same replacement in test. Feature engineering fills missing train service values with `others` and replaces test `"None"` with `others`, but does not show a matching test `'-'` replacement. Do not claim service handling is symmetric; preserve and document this behavior unless a focused change fixes the train/test mapping together.

## Feature-processing invariants

- Correlation-based feature selection is computed from training data with an absolute Pearson threshold above `0.95`; the selected columns are data-dependent. Do not hard-code the recorded example list as universal.
- `network_bytes = sbytes + dbytes`.
- The notebook replaces `dur`, `sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `stcpb`, `dtcpb`, `smeansz`, `dmeansz`, `sjit`, `djit`, and `network_bytes` with `_log1p` columns.
- Numeric scaling and categorical encoders are fit on training data and reused on test data. Categorical columns are `proto`, `service`, and `state`.
- The prepared test CSV already has column names. Do not assign the original unfiltered `saved_dict['columns']` list after dropping columns; that can cause a pandas length-mismatch error.
- Outlier analysis in feature engineering is diagnostic; do not claim or implement row removal unless explicitly requested and justified.

## Editing and validation

- Read the current notebook cells and outputs before editing; notebook outputs may be stale or may include historical errors. Prefer source code over stored outputs when describing intended behavior.
- Preserve unrelated user edits, generated local data, and existing notebooks. Avoid broad notebook reformatting/output clearing.
- Update this file when stable project workflow or artifact contracts change, and update `README.md` separately for human-facing explanations. The model-input artifacts use compressed CSR matrices plus metadata so storage/loading is sparse without changing features.
- Validate notebook JSON and syntax after editing. For behavioral changes, use focused fixtures or run the affected cells when the required local datasets and kernel are available. Do not claim a full workflow passed unless it was run end-to-end.
- There is no documented root build/test command or dependency lockfile. Do not install dependencies unless validation fails specifically because a package is missing.
