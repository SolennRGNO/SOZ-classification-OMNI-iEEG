# Feature-based seizure onset zone classification using the OMNI-iEEG dataset

## Project overview

This research internship at Université de Montréal, Coco Lab, investigates whether quantitative features of intracranial EEG can distinguish channels labelled as part of the **seizure onset zone (SOZ)** from a carefully defined non-SOZ comparison group.

The repository brings together recording inventories, preprocessing, four feature families, cohort-specific dataset construction, and participant-grouped machine learning. Its contribution is a code-based framework for comparing conventional signal descriptors with complexity measures under explicit cohort, label, and evaluation rules. It contains scientific implementations and workflow integration, rather than evidence that any particular model performs best. No clinical recordings or performance results are distributed here.

## Scientific background and objectives

Epilepsy is a brain disorder involving a persistent predisposition to seizures. Drug-resistant epilepsy refers to failure of two appropriately chosen, tolerated, and adequately used antiseizure medication regimens to achieve sustained seizure freedom; see the [ILAE definition](https://www.ilae.org/files/ilaeGuideline/Definition-of-Drug-Resistant-Epilepsy-2009-1528-1167.2009.02397.pdf). Some patients undergo a presurgical investigation to locate the regions involved in their seizures.

**Intracranial electroencephalography (iEEG)** records electrical activity through implanted electrodes. Electrocorticography (**ECoG**) uses electrodes on the brain surface; stereoelectroencephalography (**SEEG**) uses depth electrodes. Each channel samples activity relative to a recording reference, within the limited coverage of the implant.

The SOZ is the region where clinicians identify the earliest seizure activity in the available recordings. Channel classification asks whether numerical signal summaries can recover this annotation. It does not identify all tissue necessary for seizure generation, and a non-SOZ label does not establish that tissue is healthy.


## Data and cohort definitions

[OMNI-iEEG](https://omni-ieeg.github.io/omni-ieeg/) harmonizes intracranial recordings and clinical metadata from multiple sources. This project uses selected source cohorts within that resource, not its complete population or every upstream benchmark task. Recordings, channel annotations, resection information, and surgical outcomes must be obtained independently under the applicable access conditions.

| Cohort | Construction and recording state | ML input policy |
|---|---|---|
| HUP/UCLA | HUP denotes the Hospital of the University of Pennsylvania; UCLA denotes the University of California, Los Angeles. The technical feature cohort requires original sampling at least 1 kHz and a known recording state. It retains outcome as metadata rather than filtering on it. | Each experiment selects exactly one state and participants with a recognized seizure-free surgical outcome. Valid epochs are pooled across eligible recordings before channel-level aggregation. |
| Detroit | A separate OpenIEEG selection excludes the UCLA entry. Upstream rules include Detroit identity, sampling frequency, seizure-free outcome, clinical ECoG, usable channel metadata and labels, sufficient duration, sleep recordings, and preprocessing availability. | The Detroit adapter consumes already aggregated participant/channel tables and enforces a frozen cohort/schema contract. It does not repeat HUP/UCLA epoch pooling. |

**Ictal** recordings contain seizure activity; **interictal** recordings occur between seizures and need not be normal. A recording task such as sleep is distinct from its canonical recording state. HUP state is derived from task metadata; UCLA is explicitly assigned interictal in this implementation. Detroit's selected sleep recordings are assigned interictal. 

Supervised positives are explicitly SOZ-positive channels. High-confidence negatives require SOZ-negative and unresected channels in seizure-free participants; ambiguous or missing labels are excluded from supervised ML. Surgical outcome defines the retrospective comparison and is not the prediction target. Stored labels are `1`, `0`, or missing for ambiguous/unlabelled channels; report terminology such as “pathological” does not establish tissue pathology.


## Methodology overview

```text
iEEG recordings → preprocessing → epochs → feature extraction → aggregation → ML → evaluation
```

1. **Inventory and selection:** inspect recording headers and metadata, document exclusions, and freeze cohort manifests. Stable input identities and ordering keep later array jobs tied to the intended recordings.
2. **Preprocessing:** retain usable intracranial channels, remove channels marked bad, apply the reviewed site-specific notch configuration, and resample eligible signals to 1 kHz. The current `Preprocessed1kHzNoBandpass` protocol adds no broad band-pass filter. Configured notches follow the 60 Hz harmonic family through 480 Hz, subject to each recording's original Nyquist limit.
3. **Epochs:** split into complete, non-overlapping eight-second windows of 8,000 samples. Incomplete trailing windows are not padded. Fixed duration and sampling make feature definitions comparable across inputs.
4. **Features and QC:** compute each family independently on the same frozen cohort-specific FIF inputs. Reject non-finite, flat, or near-flat channel-epochs without automatically discarding other channels in the window. Extraction failures and invalid feature values must be inspected before assembly.
5. **Aggregation:** assemblers produce epoch tables and recording/channel summaries using mean, sample standard deviation, and median. HUP/UCLA ML builders instead pool epoch rows across recordings first, avoiding an average of recording averages. Detroit requires its separate preaggregated input contract.
6. **Learning and evaluation:** train on participant/channel observations and evaluate predictions for participants excluded from fitting and model selection.

FIF stores signal derivatives, Parquet stores feature tables, CSV/JSON carry audits and provenance, and YAML defines execution graphs. Generated artifacts are not supplied. Keep both cohorts separate and use a distinct derivatives root for each feature family. No feature selection occurs during extraction or assembly.

## Extracted features

| Family | Implemented descriptors | Scientific motivation |
|---|---|---|
| [Time domain](scripts/features/time_domain.py) | 12 measures: mean, median, variance, standard deviation, root mean square amplitude, line length, zero-crossing rate, Hjorth activity/mobility/complexity, skewness, and kurtosis. | Describe amplitude, waveform variation, and distribution shape without reducing the signal to frequency bands. |
| [Frequency domain](scripts/features/frequency_domain.py) | Normalized spectral entropy and peak frequency from the Welch power spectral density over 0.5–450 Hz. | Describe how concentrated or dispersed power is and where its strongest spectral component lies. |
| [Relative bandpower](scripts/features/relative_bandpower.py) | 18 conventional/extended band fractions and total-power measures, with normalization ranges extending to 150 or 450 Hz. | Compare the distribution of power across frequency ranges while distinguishing relative fractions from absolute power. These summaries do not detect individual high-frequency oscillation events. |
| [Complexity, entropy, and fractal features](scripts/features/complexity.py) | 14 measures: approximate, sample, permutation, spectral, singular-value-decomposition and multiscale entropy; Lempel–Ziv complexity; Katz, Petrosian and Higuchi fractal dimensions; Hjorth mobility/complexity; zero-crossing rate; and a FOOOF aperiodic exponent stored as `aperiodic_slope`. | Describe regularity, pattern diversity, waveform structure, and the non-oscillatory spectral background. Higher complexity does not inherently mean more pathological activity. |

Some descriptors occur in multiple families. They should not be interpreted as independent new measurements. The conventional model contains 32 epoch features aggregated into 96 predictors; complexity remains a separate 14-feature family, yielding 42 predictors when mean, standard deviation, and median are used.

## Machine learning and evaluation

| Approach | Actual implementation |
|---|---|
| Conventional logistic baseline | Fixed L2 logistic regression on the 96 conventional predictors in [logistic_baseline.py](scripts/ml/logistic_baseline.py). |
| Complexity logistic models | [logistic_complexity.py](scripts/ml/logistic_complexity.py) provides all-feature L2 (`b`), fixed-size mRMR selection followed by L2 (`c1`), inner-validation selection of subset size (`c1_nested`), and nested Elastic Net (`c2`). |
| Joint selection and threshold tuning | [logistic_complexity_tuned.py](scripts/ml/logistic_complexity_tuned.py) jointly tunes mRMR subset size and L2 regularization, then selects a threshold from inner out-of-fold probabilities. |
| Random forest | [random_forest_complexity.py](scripts/ml/random_forest_complexity.py) tunes forest hyperparameters on all 42 complexity predictors, without scaling or feature selection before the forest. |

Minimum-redundancy maximum-relevance (**mRMR**) selects predictors with high mutual information with the target and low redundancy with selected predictors. The [local selector](scripts/ml/feature_selection.py) implements a greedy mutual-information-difference criterion using scikit-learn estimators; it does not require a separate mRMR package. Elastic Net combines L1 and L2 penalties, with regularization strength and mixing ratio tuned within training data.

The principal evaluators use five outer participant-grouped folds; nested approaches use three grouped inner folds. Keeping every channel from a participant together prevents the model from being evaluated on another channel from a person it already saw during fitting. Logistic scaling, participant/class training weights, feature selection, hyperparameter search, and threshold selection are fitted within their appropriate training partitions. Training weights account for unequal channel counts per participant and class imbalance. Predictors must be finite; the evaluators do not silently impute missing values.

Saved outputs include held-out predictions, participant fold assignments, metrics, run summaries, and model-specific selection or coefficient/importance artifacts. AUROC measures ranking across thresholds; average precision summarizes precision–recall performance and depends on class prevalence. Sensitivity, specificity, precision, F1, and balanced accuracy describe thresholded predictions. Participant-cluster bootstrap intervals account for channels sharing a participant. Fold summaries and metrics pooled over out-of-fold predictions are distinct summaries, not interchangeable estimates.

The [cross-modality module](scripts/ml/crossmodality.py) is exploratory. Modality and hospital site are confounded, so transfer performance cannot isolate a modality effect. None of these protocols establishes prospective clinical validity.

## Repository organization

| Path | Role |
|---|---|
| [scripts/preprocessing/](scripts/preprocessing) | Recording inventories, metadata-based cohort selection, manifests, raw spectral QC, and execution summaries. |
| [scripts/features/](scripts/features) | Scientific feature calculations, shared label/artifact handling, cohort assemblers, and QC reports. |
| [scripts/ml/](scripts/ml) | Cohort-specific dataset builders, feature selection, estimators, evaluation, and presentation. |
| [Neurodags/](Neurodags) | Project YAML graphs and Python node adapters connecting scientific code to the external NeuroDAGs engine; current preprocessing and feature workflows, reviewed line-noise configurations, shared inventory dataset definitions, and a 1 kHz synthetic fixture generator. |
| [cluster/](cluster) | Institution-dependent Bash/SLURM stage and submission wrappers. These orchestrate execution; they do not replace the scientific implementations. |
| [experiments/](experiments) | Exploratory multiscale-entropy scale comparison, separate from the principal workflows. |
| [requirements.txt](requirements.txt) | Pinned Python 3.11 scientific and workflow dependencies. |

## Where to start / Using the code

### Install the Python dependencies

Use Python **3.11** with pip and virtual-environment support. From the repository root in a POSIX shell:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements include MNE-Python, NumPy/SciPy, pandas/PyArrow, plotting, scikit-learn, AntroPy, NeuroKit2, FOOOF, PyYAML, NeuroDAGs, and xarray/NetCDF support. Exact scientific/workflow versions and compatibility pins were preserved from the original Python 3.11 specifications, which reported a consistent installed Python 3.11.15 environment. 


### Choose an entry point

| Research goal | Existing code to inspect or use |
|---|---|
| Inspect or prepare recordings | [build_inventory.py](scripts/preprocessing/build_inventory.py), [run_raw_line_noise_qc.py](scripts/preprocessing/run_raw_line_noise_qc.py), and preprocessing graphs [HUP/UCLA](Neurodags/pipeline_hup_ucla_1khz.yml) / [OpenIEEG](Neurodags/pipeline_openieeg_1khz.yml). |
| Build feature cohorts or manifests | [build_nonictal_feature_cohort.py](scripts/preprocessing/build_nonictal_feature_cohort.py), [build_subject_manifests.py](scripts/preprocessing/build_subject_manifests.py), [build_hup_ucla_feature_manifest.py](scripts/features/build_hup_ucla_feature_manifest.py), and [build_detroit_feature_manifest.py](scripts/features/build_detroit_feature_manifest.py). |
| Extract time-domain features | [Scientific core](scripts/features/time_domain.py), [HUP/UCLA graph](Neurodags/pipeline_hup_ucla_time_domain.yml), [Detroit graph](Neurodags/pipeline_detroit_time_domain.yml). |
| Extract frequency-domain features | [Scientific core](scripts/features/frequency_domain.py), [HUP/UCLA graph](Neurodags/pipeline_hup_ucla_frequency_domain.yml), [Detroit graph](Neurodags/pipeline_detroit_frequency_domain.yml). |
| Extract relative bandpower | [Scientific core](scripts/features/relative_bandpower.py), [HUP/UCLA graph](Neurodags/pipeline_hup_ucla_relative_bandpower.yml), [Detroit graph](Neurodags/pipeline_detroit_relative_bandpower.yml). |
| Extract complexity features | [Scientific core](scripts/features/complexity.py), [HUP/UCLA graph](Neurodags/pipeline_hup_ucla_complexity.yml), [Detroit graph](Neurodags/pipeline_detroit_complexity.yml). |
| Assemble extracted artifacts | [Time domain](scripts/features/assemble_time_domain_cohort.py), [frequency domain](scripts/features/assemble_frequency_domain_cohort.py), [relative bandpower](scripts/features/assemble_relative_bandpower_cohort.py), and [complexity](scripts/features/assemble_complexity_cohort.py) assemblers. |
| Build ML datasets | HUP/UCLA [conventional builder](scripts/ml/dataset.py) and [complexity builder](scripts/ml/complexity_dataset.py); [Detroit adapter](scripts/ml/detroit_dataset.py). |
| Run classification experiments | [Conventional logistic baseline](scripts/ml/logistic_baseline.py), [complexity logistic variants](scripts/ml/logistic_complexity.py), [tuned logistic](scripts/ml/logistic_complexity_tuned.py), [random forest](scripts/ml/random_forest_complexity.py), and [Detroit experiment dispatcher](scripts/ml/run_detroit_ml.py). |

The feature cores expose Python functions used by the node adapters; running a core file is not a general extraction command. Feature graphs require **generated cohort-specific dataset configurations**, matching FIF files and preprocessing metadata, and audited channel labels. The manifest builders require `--cohort-csv`, repeatable `--preprocessed-root`, `--channel-labels-csv`, a new `--output-directory`, and one distinct `--<family>-derivatives-path` for each family. They freeze input ordering and refuse to replace an existing manifest directory. Several dataset configurations referenced by the graphs are generated inputs, not checked-in ready-to-run files.

The cohort selector has separate `--profile hup-ucla` and `--profile openieeg-detroit` modes. Detroit additionally requires `--data-root` and `--derivatives-root`; selecting the profile is a methodological choice, not just a naming change.

### Representative commands and their prerequisites

These examples match the checked-in argparse interfaces but were not executed. Run from the repository root after installing dependencies. Set each uppercase shell variable to an appropriate absolute input or output path on your system; these variables refer to external/generated artifacts, not files supplied here. Use new ML output directories, because model and dataset stages refuse to overwrite existing runs.

**1. Inventory HUP/UCLA recordings.** `DATA_ROOT` must contain the expected OMNI-iEEG participant layout, EDF recordings, and metadata sidecars. The script relocates the dataset patterns in [datasets2.yml](Neurodags/datasets2.yml) beneath that root. `--check-edf-header` reads headers without loading the full signal and is required to determine sampling-frequency eligibility.

```bash
python scripts/preprocessing/build_inventory.py \
  --data-root "$DATA_ROOT" --datasets hup ucla \
  --check-edf-header --output-dir "$INVENTORY_OUT"
```

Outputs include recording, eligible/excluded, sampling-frequency and participant-summary CSV tables, plus an HTML inventory. This inventories inputs; it does not preprocess them or apply the final ML cohort rule.

**2. Assemble a completed time-domain extraction.** `TIME_DERIVATIVES` must contain one completed NeuroDAGs feature Parquet and QC JSON artifact per recording in `COHORT_CSV`. This stage checks completeness and invalid feature values before writing combined epoch/channel Parquets, a QC CSV, and a JSON summary.

```bash
python scripts/features/assemble_time_domain_cohort.py \
  --derivatives-root "$TIME_DERIVATIVES" --cohort-csv "$COHORT_CSV" \
  --output-dir "$ASSEMBLY_OUT" --cohort-name hup_ucla --repository .
```

Use `--cohort-name detroit` for a separately extracted Detroit cohort. This changes cohort-specific artifact names; it does not perform participant-level pooling for Detroit ML.

**3. Run the conventional logistic baseline.** First build a compatible dataset with `python -m scripts.ml.dataset`: supply `--task interictal` or `--task ictal`, the three family inputs (`--frequency-domain-epochs`, `--relative-bandpower-epochs`, `--time-domain-epochs`), `--cohort-csv`, `--channel-labels-csv`, `--output-dir`, and `--repository .`. Supply explicit paths to avoid institutional defaults. Inputs must include the corresponding upstream QC/run-summary artifacts. The builder writes participant/channel Parquet and CSV tables, a feature-list JSON, and audit/provenance outputs.

Set `ML_TABLE` and `FEATURE_LIST` to that builder's Parquet and feature-list JSON outputs. The following fits the fixed conventional baseline and writes out-of-fold predictions, fold audits, metrics, coefficients, and a run summary to `MODEL_OUT`:

```bash
python -m scripts.ml.logistic_baseline \
  --dataset "$ML_TABLE" --feature-list "$FEATURE_LIST" \
  --output-dir "$MODEL_OUT" --repository .
```

`--n-bootstrap` controls participant-cluster bootstrap resampling; its default is 5,000, so this is a scientific computation rather than a lightweight smoke command. Complexity dataset construction separately requires an explicit `--aggregations` choice; the principal complexity models expect mean, standard deviation, and median. Detroit ML requires unique participant/channel inputs matching its frozen contract, rather than substituting recording-level assembler outputs.

## Limitations and data availability

Running the complete pipeline requires access to the OMNI-iEEG data and appropriate computing resources. HPC scripts were developed for the Fir cluster and require configuration before use in other environments.

The study relies on retrospective clinical annotations and selected patient cohorts. SOZ labels may be uncertain, electrode coverage is limited, and recording conditions or differences between institutions may influence the extracted features. Consequently, the results should be interpreted as exploratory research rather than clinically validated predictions.

## References and acknowledgments

- **OMNI-iEEG:** Duan, Zhang, and colleagues, *Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research* (ICLR 2026). See the [paper](https://arxiv.org/abs/2602.16072), [project page](https://omni-ieeg.github.io/omni-ieeg/), and [upstream organization](https://github.com/Omni-iEEG). This internship code is distinct from the upstream dataset and benchmark contribution.
- **NeuroDAGs:** independently developed external software by **Yorguin-José Mantilla-Ramos**; see its [upstream repository](https://github.com/yjmantilla/neurodags) and [package record](https://pypi.org/project/neurodags/). Framework developed used in this project to implement the preprocessing and feature extraction pipelines.
- **Research context:** This project was developed during a two-month research internship at Coco Lab, Université de Montréal, under the supervision of Pr. Karim Jerbi.
- **License:** This project is licensed under the MIT License. See [LICENCE](LICENCE).
