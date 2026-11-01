# IDS Team Guide

## What this guide is

This file maps the existing notebooks and the planned IDS app/source files to
the teams that will work on them. The Python files under `src/` are **comment-
only placeholders**: they explain what a team should implement, but contain no
working application, PCAP parser, or model code.

The app team may develop its UI independently with temporary local mocks. Those
mocks must be clearly marked as synthetic and replaced by the real PCAP and
model-team components before the app is used to classify traffic.

## Proposed project structure

Keep the application and its components under `src/`. The parser's planned
location is explicitly `src/pcap_parser.py`. Training and analysis remain in
the existing notebooks unless the teams later agree to move them.

```text
Network-Intrusion-Detection-System/
├── src/
│   ├── __init__.py
│   ├── app.py                 # Streamlit app and user interaction
│   ├── pcap_parser.py         # PCAP/PCAPNG -> packets/flows; PCAP team
│   ├── preprocessing.py       # flow-row schema checks and model transforms
│   ├── models.py              # load trusted fitted model and predict
│   ├── evaluation.py          # labeled-data metrics; no model fitting here
│   ├── run_pipeline.py        # optional app/integration entry-point helpers
│   └── utils.py               # small shared helpers only
├── notebooks/
│   ├── About Dataset and It's Cleaning.ipynb
│   ├── Exploratory Data Analysis.ipynb
│   ├── Feature_Engineering_and_Data_Preparation.ipynb
│   ├── ML Models and Results.ipynb
│   ├── Prediction of Raw Data.ipynb
│   ├── final_ipynb            # local prepared-data metadata, not a notebook
│   └── .final_ipynb/          # local generated matrices, preprocessors, models
├── data/                      # local datasets and captures; do not commit
├── requirements.txt
├── README.md
└── TEAM.md
```

This is a small-project starting point, not a requirement to add more layers.
Teams can reorganize it later when there is a concrete need. Do not create a
second root-level app or duplicate transformations across modules.

## Team ownership

| Team | Main files | Responsibilities |
|---|---|---|
| App development | `src/app.py`; coordinate `src/run_pipeline.py` | Build the Streamlit interface for capture upload, prediction results, understandable errors, and a separate labeled-data evaluation screen. During independent work, use local synthetic mocks and label them as demos. Integrate the parser and model team's real interfaces later. |
| PCAP-to-rows | `src/pcap_parser.py`; coordinate `src/preprocessing.py` | Parse PCAP/PCAPNG offline, group packets into flows, and document direction, timeout, counting, units, missing values, and unsupported traffic. Return flow rows under the agreed contract. Do not invent unavailable model features. |
| Models | `notebooks/Feature_Engineering_and_Data_Preparation.ipynb`, `notebooks/ML Models and Results.ipynb`, `notebooks/Prediction of Raw Data.ipynb`, `src/models.py`, `src/evaluation.py`, `src/preprocessing.py` | Own train-fitted transformations, model selection/training, compatible inference artifacts, and evaluation. Keep the held-out test data out of fitting and tuning. Provide the app an inference interface that reuses the saved fitted objects. |
| Shared | `src/__init__.py`, `src/utils.py`, `requirements.txt`, `TEAM.md` | Agree interfaces and dependencies; integrate in the same repository; update documentation when contracts change. |

## `src/` file guide

All files below are currently comment-only instructions. Teams should add code
only when they take ownership of the relevant file.

| File | Team | What belongs there |
|---|---|---|
| [`src/__init__.py`](./src/__init__.py) | Shared | Keep the source package marker minimal. |
| [`src/app.py`](./src/app.py) | App | Streamlit UI: upload, page flow, results, errors, and evaluation-page interaction. Keep model and packet parsing details out of the UI. |
| [`src/pcap_parser.py`](./src/pcap_parser.py) | PCAP-to-rows | The actual future PCAP/PCAPNG parser location. Parse packets and build/return flow records through an agreed interface. Treat uploaded captures as untrusted; parse offline and surface unsupported data. |
| [`src/preprocessing.py`](./src/preprocessing.py) | Models + PCAP coordination | Validate flow input columns and apply model-owned fitted preprocessing. Keep field mapping/semantics coordinated with the PCAP team. Never refit encoders/scalers on uploaded or test data. |
| [`src/models.py`](./src/models.py) | Models | Load trusted fitted artifacts and expose prediction for validated flow rows. Do not retrain from the app. Preserve the exact selected feature order. |
| [`src/evaluation.py`](./src/evaluation.py) | Models + app | Calculate metrics on labeled evaluation inputs. Use scores for ROC-AUC and predicted classes for class metrics. Never train/tune on the held-out test labels. |
| [`src/run_pipeline.py`](./src/run_pipeline.py) | App + integration | Optional small orchestration between upload, parser, preprocessing, model, and result display. Do not duplicate the responsibilities of the component files. |
| [`src/utils.py`](./src/utils.py) | Shared | Small helpers genuinely shared by multiple modules. Avoid turning this into a miscellaneous module. |

## Notebook guide: existing data/model workflow

The notebooks are part of the project and should remain the source of truth for
the current training workflow until the model team intentionally changes it.
Run them with `notebooks/` as the working directory.

| Notebook/file | Team | Purpose and app-team relevance |
|---|---|---|
| [`notebooks/About Dataset and It's Cleaning.ipynb`](./notebooks/About%20Dataset%20and%20It's%20Cleaning.ipynb) | Models/data | Streams the four UNSW-NB15 traffic CSV parts, takes a reproducible capped sample, cleans selected values, and creates the 70/30 train/test split. Produces prepared CSVs under `data/` and metadata in `notebooks/final_ipynb`. |
| [`notebooks/Exploratory Data Analysis.ipynb`](./notebooks/Exploratory%20Data%20Analysis.ipynb) | Data | Explores prepared data and plots distributions/correlations. It is analysis, not a runtime dependency of the app. |
| [`notebooks/Feature_Engineering_and_Data_Preparation.ipynb`](./notebooks/Feature_Engineering_and_Data_Preparation.ipynb) | Models | Creates `network_bytes`, performs train-fitted numeric/log transformations and `proto`/`service`/`state` one-hot encoding, applies fitted transformations to test data, and saves sparse feature matrices plus preprocessing artifacts under `notebooks/.final_ipynb/`. This defines the current model input contract. |
| [`notebooks/ML Models and Results.ipynb`](./notebooks/ML%20Models%20and%20Results.ipynb) | Models | Loads the persisted sparse train/test inputs, trains/tunes and evaluates classifiers, and saves fitted model artifacts under `notebooks/.final_ipynb/`. The app must use an agreed fitted model artifact, not run notebook training during a prediction request. |
| [`notebooks/Prediction of Raw Data.ipynb`](./notebooks/Prediction%20of%20Raw%20Data.ipynb) | Models + integration | Demonstrates applying saved preprocessing and the selected random forest to raw UNSW-NB15-style rows, and evaluates on the held-out prepared split. Its preprocessing is a reference for the app/model integration, not a PCAP parser. |
| [`notebooks/final_ipynb`](./notebooks/final_ipynb) | Generated local data | Pickled metadata from the cleaning notebook; despite its name, this is a local artifact, not an `.ipynb` notebook. Do not edit by hand. |
| [`notebooks/.final_ipynb/`](./notebooks/.final_ipynb/) | Generated local artifacts | Local sparse matrices, feature metadata, fitted preprocessors, and model pickles consumed by later notebooks. These are generated artifacts and are not the source code to edit. |

## Current raw model input and the PCAP compatibility issue

The current prediction workflow expects 47 raw fields per flow (not including
the training-only `label` or `attack_cat`):

```text
srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl,
sloss, dloss, service, sload, dload, spkts, dpkts, swin, dwin, stcpb, dtcpb,
smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit, stime, ltime,
sintpkt, dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports, ct_state_ttl,
ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm,
ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm
```

The fitted workflow derives `network_bytes = sbytes + dbytes`, applies numeric
transforms/scaling, and one-hot encodes `proto`, `service`, and `state`. The
feature-selected random forest uses 55 columns in its saved order.

**Matching column names is not sufficient.** PCAP-derived values must have
definitions compatible with UNSW-NB15 training. In particular, contextual
`ct_*` values and application-level `service`, `state`, HTTP, and FTP fields
may not be available from a packet capture without matching grouping rules or
visible payload. The PCAP team must document direction, timeouts, units, and
counting rules. Do not fill unknown features with guessed zeros or dataset
modes. If semantics cannot be matched, the model team must retrain and evaluate
using data generated by the chosen parser, or the app must reject/flag those
flows rather than claim a reliable prediction.

## Integration sequence

1. Agree on the parser output, raw field definitions, units, direction, errors,
   and model prediction interface.
2. PCAP team implements `src/pcap_parser.py` and demonstrates the fields it can
   calculate from a small sanitized capture.
3. Models team verifies feature compatibility, supplies trusted fitted
   artifacts, and implements inference in `src/models.py` using
   `src/preprocessing.py`.
4. App team connects these modules in `src/app.py`; local mocks used during
   development must be replaced before real predictions are presented.
5. Test a sanitized capture end-to-end. Use labeled records separately for
   evaluation, and never fit/tune on the held-out test set.

Keep captures and datasets private and out of Git. Avoid logging payloads or
private endpoint information. This is a student mini-project, not a
production intrusion-detection guarantee.
