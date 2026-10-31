# UNSW-NB15 Network Intrusion Detection

This repository is a notebook-based data-mining project using the UNSW-NB15 network traffic dataset. It explores the data, prepares a binary attack-detection dataset, and experiments with supervised classifiers. The notebooks are educational and exploratory: their saved cells and plots are not proof that the full workflow runs successfully in a fresh environment, and this repository does not provide a production intrusion-detection system.

## What the project does

The target for the main modeling workflow is `label`: `0` means normal traffic and `1` means attack traffic. The dataset also contains `attack_cat`, the attack type; the feature-engineering workflow drops it and therefore performs binary, not multiclass, classification.

The notebooks follow this general sequence:

1. Load and describe UNSW-NB15, clean selected values, and create a 70/30 train/test split.
2. Explore correlations, categorical frequencies, and normal-versus-attack feature distributions.
3. Engineer numeric and categorical features using training-fitted transformations.
4. Save the transformed train and test splits for the model notebook.
5. Tune and evaluate classifiers using the saved splits.

## Dataset and setup

The data files are not included in Git. The cleaning notebook expects these files under `data/`:

- `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, and `UNSW-NB15_4.csv`
- `NUSW-NB15_features.csv` (the spelling in the notebook is `NUSW`)

The notebook loads the four data parts without headers, uses the feature-name CSV to assign column names, and expects the UNSW-NB15 schema. See the [dataset information and cleaning notebook](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) for its source reference and feature overview.

The notebooks' relative paths expect `notebooks/` to be the working directory: data paths use `../data`, while model artifacts use `./.final_ipynb`. If you start Jupyter from the repository root, configure the notebook working directory accordingly or update all related paths consistently. Run notebook cells in order. The workflow uses Python packages including pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, tqdm, PrettyTable, and XGBoost. There is no dependency lockfile or documented test/build command; package compatibility may matter because the notebooks contain older scikit-learn APIs.

## Notebook guide

### 1. Dataset information, cleaning, and split

[About Dataset and It's Cleaning.ipynb](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) introduces IDS concepts, the UNSW-NB15 features and labels, and the project's classification objective. Its code:

- Combines the four raw CSV parts and adds column names from the features file.
- Creates a random 70/30 train/test split (`test_size=0.3`, `random_state=16`) before cleaning.
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
- Checks that train and test feature columns agree, then saves `(features, labels)` for both splits.

The fitted preprocessing objects and metadata, plus the final model inputs, are stored in `notebooks/.final_ipynb/`. In particular:

- `final_train.pkl` contains `(x_train, y_train)`.
- `final_test.pkl` contains `(x_test, y_test)`.
- Other files in that folder store the scaler, encoders, fill values, and preprocessing metadata.

Run the feature-engineering notebook from the beginning through the final save cell before opening the model notebook. The original prepared CSVs are not overwritten. Pickle files under `notebooks/` are excluded from Git; each user must generate them locally.

The current service handling is not fully symmetric: feature engineering fills missing training `service` values with `others` and replaces test values equal to `"None"` with `others`, but the cleaning notebook does not replace test `-` values with `"None"`. Verify how those test categories are handled before relying on the service feature.

### 4. Model experiments

[ML Models and Results.ipynb](./notebooks/ML%20Models%20and%20Results.ipynb) loads the train/test pickle tuples and preprocessing artifacts from `notebooks/.final_ipynb/`. It converts feature tables to sparse matrices and experiments with:

- SGD-based logistic classification and a linear SVM.
- Decision trees, random forests, and XGBoost gradient-boosted trees.
- Hyperparameter search / cross-validation, feature-importance-based experiments, and a voting ensemble.

The notebook reports AUC, F1-score, and a custom false-alarm metric, and displays ROC curves and confusion matrices. It also saves model/result artifacts in the same local artifact folder. Some operations are memory- and compute-intensive; the notebook has historical outputs containing a sparse-matrix `MemoryError`, so those outputs should not be mistaken for a successful fresh run. The false-alarm function in the notebook is defined as the average of false-positive and false-negative rates.

## Running the workflow

1. Obtain the raw dataset and feature-name CSV, and place them in `data/` using the expected filenames above.
2. Run [About Dataset and It's Cleaning.ipynb](./notebooks/About%20Dataset%20and%20It%27s%20Cleaning.ipynb) to create the prepared train/test CSVs and the metadata pickle.
3. Run [Feature_Engineering_and_Data_Preparation.ipynb](./notebooks/Feature_Engineering_and_Data_Preparation.ipynb) from top to bottom to create the model-ready pickle tuples and preprocessing objects.
4. Run [ML Models and Results.ipynb](./notebooks/ML%20Models%20and%20Results.ipynb) to load those files and train/evaluate models.

Keep the prepared test split separate: do not fit preprocessing on it or recombine it with training data for a new random split when comparing the models in the model notebook. If a data file, feature list, or preprocessing operation changes, regenerate the artifacts before modeling.

## Data and generated-file handling

Files in `data/` are ignored by Git because the raw and derived datasets are large. Pickle artifacts (`*.pkl`) anywhere under `notebooks/` are also ignored. Do not commit raw datasets, generated CSVs, or trained model binaries unless explicitly required. The `notebooks/final_ipynb` metadata pickle has no extension, so it is not covered by the `.pkl` ignore pattern.
