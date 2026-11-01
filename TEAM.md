# Team Integration Guide: PCAP to Traffic Prediction

## Goal and scope

This is a small data-mining project, so the first app should be one Streamlit
application with Python modules behind it—not a frontend, API server, database,
and deployment stack built as separate projects.

The user uploads a packet capture (`.pcap` or `.pcapng`). The app extracts
bidirectional network flows, converts each flow to a validated model-input
record, runs the fitted preprocessing and classifier, then displays one result
per flow and a file-level summary. A separate evaluation page accepts labeled
records and reports model metrics. Prediction and evaluation are separate
actions: an uploaded unlabeled capture is not training data.

```text
PCAP/PCAPNG upload
  -> offline packet parsing
  -> bidirectional flow aggregation
  -> model-schema adapter and validation
  -> saved preprocessing + classifier
  -> per-flow predictions and capture summary
```

## The critical PCAP-to-record problem

The current model is not a generic packet classifier. It was trained on the
UNSW-NB15 schema and expects the raw feature columns below before its saved
preprocessing can run. The schema includes `label` in the training dataset and
`attack_cat` as a dataset label; neither should be required or supplied for a
live prediction.

Required prediction input columns (as of the current notebook artifacts):

```text
srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl,
sloss, dloss, service, sload, dload, spkts, dpkts, swin, dwin, stcpb, dtcpb,
smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit, stime, ltime,
sintpkt, dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports, ct_state_ttl,
ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm,
ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm
```

The predictor currently drops identifiers (`srcip`, `sport`, `dstip`, `dsport`)
and `attack_cat`, removes some correlated columns, creates
`network_bytes = sbytes + dbytes`, applies the saved `log1p` transforms, then
uses the saved scaler and `proto`/`service`/`state` one-hot encoders. The model
uses the feature-importance-selected 55-column subset in the exact saved order.
The model team must publish the current artifact's schema and selected feature
order with the model bundle; do not make a second, hand-maintained feature list
in the UI.

### How the converter should build a flow record

1. Parse packets offline. For a mini-project, choose one capture parser and
   keep it behind `src/capture/parser.py`. Scapy is a reasonable pure-Python
   choice; if the team chooses CICFlowMeter or another extractor, wrap its
   output behind the same internal contract. Do not have the UI parse packets.
2. Group packets into bidirectional flows using protocol and endpoint tuples.
   Define direction consistently (for example, first-seen endpoint is source),
   and document idle/active timeout rules. A flow, not an individual packet,
   is one candidate row for this model.
3. Calculate only values whose definitions are documented and reproducible:
   timestamps/duration, forward and reverse packet and byte counts, packet-size
   means, inter-arrival intervals/jitter, observed TTLs, TCP window/sequence
   information when present, and transport protocol/state.
4. Derive higher-level fields only when their semantics are known. Examples:
   `sbytes`/`dbytes` depend on the chosen flow direction and byte-counting
   convention; `sload`/`dload` depend on whether headers and retransmissions
   count; `service`, `state`, and HTTP/FTP fields depend on protocol
   dissection and may be unavailable for encrypted traffic.
5. Treat contextual `ct_*` features as a separate requirement. They count
   activity across flows or time windows and require exact definitions,
   grouping keys, and window/reset rules. Do not set these fields to zero or
   copy dataset modes merely to satisfy the column check.
6. Emit one row per flow and validate column names, data types, units,
   categorical values, and missing-value policy against a versioned schema.
   Preserve an extraction status/reason for values that cannot be computed.
   Reject or flag unsupported records; never silently fabricate inputs.

**Important compatibility gate:** having the right column names is not enough.
The current model is trained on UNSW-NB15 records, not on this project's own
PCAP-extractor output. The converter team must compare its field definitions
and distributions with the model's training definitions. If a field—especially
a contextual `ct_*` feature, `service`, or `state`—cannot be reproduced with
matching semantics, the current model cannot be assumed valid for those rows.
The safe choices are (a) implement and verify compatible extraction, or
(b) build a labeled dataset using the chosen PCAP extractor and retrain and
evaluate the full model pipeline on that representation. Do not bridge the
gap with guessed defaults.

## Model handoff and prediction contract

The model team owns a versioned release directory containing:

```text
models/
  release/
    model.joblib                 # complete inference artifact, if pipeline is bundled
    preprocessing/               # fitted preprocessing if still separate
    metadata.json                # schema, feature order, versions, model and metrics
```

The existing notebook artifacts are currently separate pickle files under
`notebooks/.final_ipynb/`. Do not move or rename them as an isolated change.
For app integration, prefer a single sklearn `Pipeline` saved with joblib that
includes all preprocessing and the estimator; this reduces model/preprocessor
drift. If producing that pipeline requires too much refactoring for the mini
project, the first app version may load the existing trusted estimator and
preprocessing artifacts together, with metadata and a smoke test proving they
match. Pickle/joblib files must only be loaded from trusted project artifacts
and should be generated using documented compatible Python/scikit-learn
versions.

The inference interface should accept a validated DataFrame of raw flow
records and return at least:

```text
flow identifier | predicted class (normal/attack) | optional model score
```

The app must not call `fit`, `fit_transform`, or retrain when loading the
inference artifact. Only apply transformations fitted by the model workflow.
Show a model score as a score, not a calibrated probability, unless calibration
has been verified.

## Teams, ownership, and collaboration boundaries

Use separate feature branches, for example `team/capture-parser`,
`team/model-serving`, and `team/streamlit-app`. Each team should work primarily
in its owned paths below. The shared schema and integration tests are
coordinated interfaces: propose changes there and agree them before merging.
Do not fork the repository into disconnected copies; integrate branches back
into the same project so the app can import the teams' modules.

| Team | Primary ownership | Deliverables |
|---|---|---|
| **PCAP-to-rows team** | `src/capture/`, `tests/capture/`, `docs/contracts/flow-record-schema.json` | Offline parser, flow aggregation, field definitions/units, invalid-input behavior, sample fixtures, and compatibility evidence. |
| **Models team** | `notebooks/`, `src/model/`, `models/release/`, `tests/model/` | Reproducible training/evaluation, preprocessing + estimator bundle, metadata/schema contract, class/score semantics, artifact smoke test. Never fit on the held-out test split. |
| **App team** | `app.py`, `src/app/`, `src/inference/`, `src/evaluation/`, `tests/app/` | Streamlit upload and validation, prediction display, capture summary, separate labeled-data evaluation, and user-facing errors. Do not duplicate model preprocessing. |
| **Integration (all teams)** | `docs/contracts/`, `tests/integration/`, `TEAM.md` | Agree the schema and field semantics, review contract changes, run an end-to-end fixture from capture to prediction, and update this handoff. |

`data/` is for local datasets and captures and remains Git-ignored. Do not
commit user captures, raw datasets, generated test data, or model binaries
unless the team explicitly decides that a sanitized artifact is suitable for
distribution.

## Proposed project structure

```text
Network-Intrusion-Detection-System/
├── app.py                              # Streamlit entry point; thin UI layer
├── src/
│   ├── capture/
│   │   ├── parser.py                   # PCAP/PCAPNG parsing boundary
│   │   ├── flow_builder.py             # bidirectional flow aggregation
│   │   └── adapter.py                  # extracted flow -> model raw schema
│   ├── model/
│   │   ├── inference.py                # load artifact and predict; no fitting
│   │   └── artifacts.py                # trusted artifact paths and metadata checks
│   ├── app/
│   │   ├── upload.py                   # upload constraints and validation
│   │   └── views.py                    # prediction and evaluation pages
│   └── evaluation/
│       └── metrics.py                  # metrics for labeled evaluation records
├── models/
│   └── release/                        # generated model bundle; normally ignored
├── docs/
│   └── contracts/
│       └── flow-record-schema.json     # agreed field names, types, units, version
├── tests/
│   ├── capture/                        # parser and flow aggregation fixtures
│   ├── model/                          # artifact/schema/prediction smoke tests
│   ├── app/                            # input and UI-facing behavior tests
│   └── integration/                    # PCAP fixture -> flow -> prediction
├── data/                               # local captures/datasets; never commit
├── notebooks/                          # existing training and analysis workflow
├── README.md
└── TEAM.md
```

This is a proposed ownership map; folders not present yet should be created by
the team responsible for them. Keep `app.py` small and avoid committing local
artifacts into `models/release/` without an explicit distribution decision.

## Minimum integration milestones

1. **Agree the contract:** model team publishes exact feature names, types,
   units, category behavior, selected feature order, and artifact version;
   capture team records which fields it can derive and how.
2. **Prove conversion:** create a small sanitized PCAP fixture and expected
   flow rows. Test direction, timeouts, retransmissions, missing protocols,
   unsupported/encrypted traffic, and contextual counters.
3. **Prove model compatibility:** run the exact preprocessing and model on
   converter-produced rows. Compare field distributions and investigate every
   fallback. If semantics differ, pause integration and retrain/reevaluate
   with compatible data.
4. **Build the app:** implement upload, limits, progress/error reporting,
   prediction table, and downloadable result CSV. Keep evaluation of labeled
   records on a distinct page.
5. **End-to-end check:** run one test capture through parsing, schema
   validation, preprocessing, prediction, and UI result rendering. Add a smoke
   test that loads the released artifact and predicts one valid row.

## Security and practical limits

Treat capture files as untrusted input. Parse offline; impose file-size,
packet-count, and processing-time limits; avoid executing uploaded content;
do not retain uploads after processing unless the user explicitly requests it;
and avoid displaying or logging payloads and private endpoint identifiers.
The first version is a demonstration/educational tool, not a production network
security monitor or a guarantee that a flow is malicious.

