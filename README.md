# UNSW-NB15 Network Intrusion Detection

This repository is a notebook-based data-mining project using the UNSW-NB15 network traffic dataset. It explores the data, prepares a binary attack-detection dataset, and experiments with supervised classifiers. The notebooks are educational and exploratory: their saved cells and plots are not proof that the full workflow runs successfully in a fresh environment, and this repository does not provide a production intrusion-detection system.

## What the project does

The target for the main modeling workflow is `label`: `0` means normal traffic and `1` means attack traffic. The dataset also contains `attack_cat`, the attack type; the feature-engineering workflow drops it and therefore performs binary, not multiclass, classification.

The notebooks follow this general sequence:

1. Uniformly sample a configurable number of raw records across the four UNSW-NB15 traffic parts, clean selected values, and create a 70/30 train/test split.
2. Explore correlations, categorical frequencies, and normal-versus-attack feature distributions.
3. Engineer numeric and categorical features using training-fitted transformations.
4. Save the transformed train and test splits for the model notebook.
5. Tune and evaluate classifiers using the saved splits.

## Dataset and setup

The data files are not included in Git. The cleaning notebook expects these files under `data/`:

- `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, and `UNSW-NB15_4.csv`
- `NUSW-NB15_features.csv` (the spelling in the notebook is `NUSW`)
- `UNSW-NB15_LIST_EVENTS.csv` may also be present; it is an auxiliary event summary, not a traffic-record part used for modeling.

The notebook streams records from the four traffic parts and draws a reproducible uniform sample across their combined rows before preprocessing. The feature-name CSV supplies the schema; it is not a fifth record dataset. The events-list CSV has a different summary schema and is not included in the sample. See the [dataset information and cleaning notebook](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) for its source reference and feature overview.

The notebooks' relative paths expect `notebooks/` to be the working directory: data paths use `../data`, while model artifacts use `./.final_ipynb`. If you start Jupyter from the repository root, configure the notebook working directory accordingly or update all related paths consistently. Run notebook cells in order. The workflow uses Python packages including pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, tqdm, PrettyTable, and XGBoost. There is no dependency lockfile or documented test/build command; package compatibility may matter because the notebooks contain older scikit-learn APIs.

## Notebook guide

### 1. Dataset information, cleaning, and split

[About Dataset and It's Cleaning.ipynb](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) introduces IDS concepts, the UNSW-NB15 features and labels, and the project's classification objective. Its code:

- Uses the user-editable global `MAX_RAW_ROWS` (default `50_000`) to take a reproducible uniform reservoir sample across all four raw traffic CSV parts, streaming rows rather than loading the full dataset into memory. `RAW_SAMPLE_SEED` controls the sample. The 10,000-record project minimum is checked before preprocessing.
- Adds column names from the separate feature-schema file, then creates a random 70/30 train/test split (`test_size=0.3`, `random_state=16`) before cleaning. With the default cap, the split contains about 35,000 training and 15,000 test records.
- Inspects missing data and feature types; fills missing `attack_cat` values in training with `normal` and normalizes attack-category text.
- Checks the binary-looking `is_ftp_login` column and maps values greater than 1 to 1 in training.
- Replaces `-` in the training split's `service` category with the literal string `"None"` (meaning no service), not Python `None`. The cleaning cell does not apply the same replacement to the test split.
- Writes `data/train_alldata_EDA.csv` and `data/test_alldata_EDA.csv`, and saves a metadata dictionary as `notebooks/final_ipynb` (a pickle file without a `.pkl` extension).

The notebook's saved outputs show example row counts, but the actual counts depend on the input files and a fresh run. The CSVs are prepared intermediate files, not the final model features.

### 2. Exploratory analysis

[Exploratory Data Analysis.ipynb](./notebooks/Exploratory%20Data%20Analysis.ipynb) loads the prepared train and test CSVs with `keep_default_na=False`, then focuses its analysis on training data. It:

- Computes Pearson correlations and displays them in a heatmap.
- Uses sampled pairplots to inspect selected highly correlated numeric features.
- Compares attack and normal traffic using category counts and numeric distributions.
- Uses `log1p` plots to help visualize strongly skewed numeric features.
- Groups selected low-frequency categories into `others` for plotting.

The notebook creates or changes DataFrame variables in memory for analysis; it does not save those EDA changes back to the prepared CSV files. Its plots are descriptive, not model validation.

### 3. Feature engineering and model-input files

[Feature_Engineering_and_Data_Preparation.ipynb](./notebooks/Feature_Engineering_and_Data_Preparation.ipynb) reads the two prepared CSVs and uses `label` as the binary target. Its main transformations are:

- Removes numeric columns selected from the training Pearson-correlation matrix using an absolute-correlation threshold above `0.95`. The recorded output lists `sloss`, `dloss`, `dpkts`, `dwin`, `ltime`, `ct_srv_dst`, `ct_src_dport_ltm`, and `ct_dst_src_ltm`; this selection is data-dependent.
- Creates `network_bytes = sbytes + dbytes`.
- Drops `srcip`, `sport`, `dstip`, `dsport`, and `attack_cat`; separates `label` into `y_train` and `y_test`.
- Replaces each original feature in `dur`, `sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `stcpb`, `dtcpb`, `smeansz`, `dmeansz`, `sjit`, `djit`, and `network_bytes` with its corresponding `_log1p` feature.
- Fits a `StandardScaler` on the training numeric features and uses it to transform test data.
- Fits separate one-hot encoders for `proto`, `service`, and `state` on training data, then reuses those encoders for test data.
- Checks that train and test feature columns agree, then saves the numeric features as compressed sparse CSR matrices and saves the ordered feature names and labels separately.

The fitted preprocessing objects and metadata, plus the final model inputs, are stored in `notebooks/.final_ipynb/`. The final feature matrices are stored as compressed SciPy CSR `.npz` files to avoid serializing dense arrays; companion metadata pickles preserve the original feature names/order and labels:

- `final_train.npz` and `final_test.npz` contain the same numeric feature values in the same column order as the prepared feature tables.
- `final_train_meta.pkl` and `final_test_meta.pkl` contain `(feature_columns, labels)` for their respective splits.
- Other files in that folder store the scaler, encoders, fill values, and preprocessing metadata.

Run the feature-engineering notebook from the beginning through the final save cell before opening the model notebook. The model notebook loads the CSR matrices directly and uses the saved feature-name lists for reporting; it does not build a duplicate feature DataFrame. The feature selection, one-hot columns, column order, and values are not changed by this storage format. The original prepared CSVs are not overwritten. Local model artifacts under `notebooks/` are excluded from Git; each user must generate them locally.

#### Viewing sparse model inputs as a DataFrame

The modeling notebook includes `csr_to_sparse_dataframe(matrix, columns, index=None)`, which uses `pandas.DataFrame.sparse.from_spmatrix` to create a DataFrame with sparse-backed columns. For example, `csr_to_sparse_dataframe(x_train_csr[:5], train_columns)` gives a readable five-row preview while keeping zeros implicit. The helper checks that feature names and any supplied index match the matrix dimensions.

This conversion is useful for inspection, display, and pandas operations that need named columns; it is **not required for model training**. Keep the saved CSR matrices as the canonical model inputs because scikit-learn and XGBoost can consume them directly. A pandas sparse DataFrame still has per-column/index overhead, so use CSR for repeated model operations and avoid `.toarray()` on a complete split. The legacy `get_final_data` preprocessing helper also now represents its `proto`, `service`, and `state` one-hot columns as sparse DataFrame columns rather than expanding them into dense arrays.

The current service handling is not fully symmetric: feature engineering fills missing training `service` values with `others` and replaces test values equal to `"None"` with `others`, but the cleaning notebook does not replace test `-` values with `"None"`. Verify how those test categories are handled before relying on the service feature.

### 4. Model training, recorded results, and app handoff

[ML Models and Results.ipynb](./notebooks/ML%20Models%20and%20Results.ipynb) loads the saved CSR features, labels, feature names, and preprocessing objects from `notebooks/.final_ipynb/`.

#### Training data and validation

With the configured `MAX_RAW_ROWS=50_000`, the cleaning notebook samples up to 50,000 raw traffic records from the four UNSW-NB15 traffic CSVs, then makes the 70/30 train/test split. The saved model-notebook outputs show 35,000 training records and 15,000 test records. In the later manual tuning stage, the training split is divided into 28,000 train rows and 7,000 validation rows. The model inputs in those outputs have 168 encoded columns; the feature-importance experiment uses a selected 55-column subset.

The notebook tries SGD-based linear classifiers, a decision tree, a random forest, XGBoost, feature-importance experiments, and a soft-voting ensemble. The grid-search helper selects by ROC-AUC scoring; the random-forest and XGBoost tuning helper compares candidate settings against the separate validation split. Some candidate searches are sequential, tuning one parameter at a time rather than exhaustively testing every parameter combination.

The parameter settings below are the final settings written in the notebook, not a claim that every reported result came from a correctly selected tuned estimator:

| Model / experiment | Final or searched settings recorded in the notebook |
|---|---|
| `LR` | `GridSearchCV` searches `SGDClassifier(loss="log_loss")`, `penalty` `l1`/`l2`, and `alpha` from `10^-6` through `10^1`; evaluation uses `lr_clf.best_estimator_` from the final search. |
| `SVM` | `GridSearchCV` searches `SGDClassifier(loss="hinge")`, `penalty` `l1`/`l2`, and `alpha` from `10^-5` through `10^2`; evaluation uses `svm_clf.best_estimator_`. This is a linear hinge-loss SGD classifier, not `sklearn.svm.SVC`. |
| `DT` / `DT_FI` | Searches `max_depth` and `min_samples_split`, then searches `min_samples_leaf` while retaining the first-stage best parameters. The final tree and feature-importance tree use the resulting `dt_clf.best_params_`; `DT_FI` uses the selected feature subset. |
| `RF` / `RF_FI` | `criterion="gini"`, `max_depth=22`, `min_samples_split=6`, `n_estimators=300`, `n_jobs=-1`. `RF_FI` uses the selected feature subset. |
| `XGB` | `n_estimators=400`, `max_depth=12`, `learning_rate=0.1`, `colsample_bylevel=0.5`, `subsample=0.1`, `n_jobs=-1`. |
| `Voting` | Soft-voting ensemble of the configured decision tree, random forest, and XGBoost models; `n_jobs=-1`. The saved run evaluates it on the feature-selected inputs. |

#### Evaluation metrics and results

The evaluation helper now calculates ROC-AUC from continuous scores: `predict_proba()` for estimators that provide probabilities, or `decision_function()` for the hinge-loss SGD classifier. F1 and confusion matrices continue to use predicted class labels. The additional “Mean FPR/FNR” metric is the average of false-positive and false-negative rates; it is a balanced error measure, not the conventional false-alarm rate. The saved results dictionary retains the legacy key `far` for compatibility.

The notebook contained previously saved scores from the old hard-label AUC calculation and mismatched LR/SVM/tree estimators. Those evaluation and summary outputs have been cleared so they are not mistaken for results from the corrected code. **No corrected scores are available until the model notebook is rerun.** Run it top-to-bottom to regenerate the test metrics and overwrite the model artifacts described below.

#### Where and how models are saved

The model notebook sets `file_path="./.final_ipynb"` and uses Python `pickle.dump` to serialize fitted estimators as `.pkl` files. When run from the documented `notebooks/` working directory, the model files are saved under `notebooks/.final_ipynb/`:

| File | Contents |
|---|---|
| `lr_clf.pkl` | Fitted estimator saved after the LR experiment (see estimator caveat above). |
| `svm_clf.pkl` | Fitted estimator saved after the SVM experiment (see estimator caveat above). |
| `dt_clf.pkl` | Fitted decision-tree estimator. |
| `rf_best_clf.pkl` | Fitted random-forest estimator. |
| `xgb_clf.pkl` | Fitted XGBoost estimator. |
| `dt_clf_imp.pkl`, `rf_clf_imp.pkl` | Fitted decision-tree and random-forest estimators for the feature-importance experiment. |
| `v_clf.pkl` | Fitted soft-voting ensemble. |
| `final_result_dict.pkl` | Aggregated result values. |

Feature inputs and preprocessing artifacts are also local to `notebooks/.final_ipynb/`: `final_train.npz` and `final_test.npz` are compressed CSR matrices; `final_train_meta.pkl` and `final_test_meta.pkl` store `(feature_columns, labels)`; `scaler.pkl`, `ohe_proto.pkl`, `ohe_service.pkl`, `ohe_state.pkl`, and the other preprocessing metadata files preserve the fitted transformations and feature-selection information. Some notebook cells write `result_dict.pkl` relative to the current working directory instead of `file_path`.

These files are ignored by Git and are not included in a clone. The serialization format and paths are unchanged by the evaluation fixes; rerunning the model notebook overwrites the same estimator pickle names with the corrected fitted estimators. Pickle files are Python/library-version-sensitive and are not a standalone app-serving package. An app must apply the same cleaning, engineered columns, fitted scaler, one-hot encoders, feature order, and (for feature-selected models) selected feature indices before calling `predict`. Do not refit preprocessing on app data. Keep the saved artifacts private to the project and load only trusted pickle files.

#### Team handoff summary

```text
Raw UNSW-NB15 traffic CSVs (four parts; sample up to 50,000 records)
  -> clean selected values and split 70% train / 30% test
  -> drop identifiers and attack_cat; use label as the binary target
  -> add network_bytes = sbytes + dbytes
  -> replace selected skewed numeric features with log1p versions
  -> fit numeric scaler and proto/service/state one-hot encoders on training data
  -> save aligned 168-column sparse CSR matrices, feature names, labels, and fitted preprocessors
  -> tune/evaluate classifiers; feature-selection experiments use 55 columns
  -> pickle model estimators and result metadata under notebooks/.final_ipynb/
  -> app applies the exact same preprocessing and feature order before inference
```

The test split must remain separate: do not fit preprocessing on it or recombine it with training data for a new random split. Before app integration, rerun the corrected evaluation, review the regenerated scores, and record the software versions used to generate the pickle artifacts.

### 5. Raw-record prediction and held-out evaluation

[Prediction of Raw Data.ipynb](./notebooks/Prediction%20of%20Raw%20Data.ipynb) applies the saved random-forest feature-importance model to raw UNSW-NB15 records. It loads the model and its matching preprocessing artifacts from `./.final_ipynb/`, and defaults to the current held-out prepared split at `../data/test_alldata_EDA.csv`.

Run this notebook from the `notebooks/` working directory, after running the cleaning, feature-engineering, and model notebooks in order. It expects:

- `notebooks/.final_ipynb/rf_clf_imp.pkl`, fitted to the feature-importance-selected inputs.
- `saved_dict.pkl`, `mode_dict.pkl`, `scaler.pkl`, `ohe_proto.pkl`, `ohe_service.pkl`, and `ohe_state.pkl` in the same folder.
- `final_train_meta.pkl` in that folder, which supplies the exact encoded feature names/order; the selected indices and model input width are checked on load.
- `data/test_alldata_EDA.csv`, containing the current held-out rows and a `label` column for evaluation.

The notebook separates `label` from the prediction input and ignores `attack_cat`, which is not available for a new live record. `final_fun_1` accepts a pandas DataFrame or a named Series with all 47 raw predictor fields from the prepared UNSW-NB15 schema; it also tolerates the prepared test CSV's extra `label` and `attack_cat` columns. It applies the saved fill values, `network_bytes`, log transforms, training-fitted scaler, and three fitted encoders. As in the feature-engineering test path, a no-service marker is mapped to the trained `others` category. One-hot features and final selected model inputs remain sparse CSR; the model continues to use the same 55 saved features in the same selected order. Unknown categories are handled by the fitted encoders' `handle_unknown="ignore"` behavior.

`final_fun_2` evaluates the held-out rows using class probabilities for standard ROC-AUC; F1 and confusion-derived values use predicted classes. It reports ROC-AUC, F1, and the mean of false-positive and false-negative rates (not conventional FPR-only false-alarm rate), and plots the ROC and confusion/precision/recall summaries. The notebook's old output was cleared; run all cells to produce metrics for the current 50,000-record sample. For live inference, call `final_fun_1` with raw records and do not pass an expected label.

With the current local model artifacts and 50,000-record sample, the 15,000-row held-out evaluation produced ROC-AUC `0.999572`, F1 `0.968443`, and mean FPR/FNR `0.015545`. The raw-to-model preprocessing was also checked against the saved selected CSR test features on 250 rows and matched exactly. These are results for the current local artifacts; retraining can change them.

This notebook does not load the old dense `final_test.pkl`, which was generated from a much larger historical dataset. It uses the current prepared test CSV for evaluation, so changing `MAX_RAW_ROWS` or the sample seed requires rerunning cleaning, feature engineering, and model training before prediction.

## Running the workflow

1. Open the repository in VS Code and select a Python kernel with the notebook dependencies installed.
2. Verify the notebook's working directory is `notebooks/` (for example, run `import os; print(os.getcwd())` in a cell). The notebooks use relative paths based on this directory.
3. If needed, obtain the raw dataset files listed above. Set `MAX_RAW_ROWS` in the first code cell of [About Dataset and It's Cleaning.ipynb](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) (default `50_000`), then run that notebook from top to bottom. This recreates `train_alldata_EDA.csv`, `test_alldata_EDA.csv`, and the metadata from the newly sampled raw records; do this whenever the cap or seed changes.
4. Run [Feature_Engineering_and_Data_Preparation.ipynb](./notebooks/Feature_Engineering_and_Data_Preparation.ipynb) from top to bottom to create compressed CSR feature matrices (`final_train.npz`, `final_test.npz`), companion feature/label metadata pickles, and the preprocessing objects from the sampled train/test files.
5. Confirm `.final_ipynb/final_train.npz` and `.final_ipynb/final_test.npz` were created, then run [ML Models and Results.ipynb](./notebooks/ML%20Models%20and%20Results.ipynb). Model training and cross-validation can still be compute-intensive.
6. Run [Prediction of Raw Data.ipynb](./notebooks/Prediction%20of%20Raw%20Data.ipynb) from top to bottom to evaluate the selected-feature random forest on the current held-out prepared CSV or to call `final_fun_1` with new raw records.

Keep the prepared test split separate: do not fit preprocessing on it or recombine it with training data for a new random split when comparing the models in the model notebook. If a data file, feature list, or preprocessing operation changes, regenerate the artifacts before modeling.

## Data and generated-file handling

Files in `data/` are ignored by Git because the raw and derived datasets are large. Pickle artifacts (`*.pkl`) anywhere under `notebooks/` are also ignored. Do not commit raw datasets, generated CSVs, or trained model binaries unless explicitly required. The `notebooks/final_ipynb` metadata pickle has no extension, so it is not covered by the `.pkl` ignore pattern.
