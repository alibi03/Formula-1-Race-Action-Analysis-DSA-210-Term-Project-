# DSA210 Term Project — Formula 1 "Race Action" Analysis

## Motivation

Some F1 races have lots of position changes, others have almost none. This project builds a race level dataset and analyses what factors make a race "high action" — defined as having a high mean absolute position change from grid to finish.

## Research Questions

- What race conditions (weather, circuit type, pit activity) are linked to more position changes?
- Do wet races produce significantly more action than dry ones?
- Does circuit layout (street vs permanent) affect how much shuffling occurs?

## Data Sources
FastF1(https://theoehrly.github.io/Fast-F1/)
Manual circuit table (in notebook) FIA
Scope: seasons 2018–2024, one row per race (~150–170 races total).

## How to Reproduce the Analysis

### 1. Clone the repository
### 2. Create a virtual environment and install dependencies
### 3. Run the notebooks

**Data & EDA:** `notebooks/01_data_eda_hypothesis.ipynb` — run all cells top to bottom. The first run downloads F1 data from the FastF1 servers and caches it locally under `data/raw/fastf1_cache/`; later runs load from cache and complete in seconds.

That notebook will:
- Build a race-level dataset and save it to `data/processed/race_level_data.csv`
- Produce EDA visualisations saved to `reports/figures/`
- Print hypothesis test results with effect sizes and multiple-testing corrections

**Machine learning:** After `race_level_data.csv` exists, run `notebooks/02_ml_models.ipynb`. It trains leak-free Random Forest models (classification for high-action races and regression for mean absolute position change) using shared code under `src/`.

## Methods

-Primary metric: mean absolute position change per classified driver (|grid − finish|)
-EDA: histograms, boxplots, scatter plots, Pearson correlation heatmap
-Hypothesis tests:

| Wet vs Dry race action | Independent-samples t-test |
| Street vs Permanent circuit action | Independent-samples t-test |
| Temperature vs action | Pearson correlation |
| Pit stops vs action | Pearson correlation |

**ML (post–EDA):** Random Forest classifier (ROC-AUC / stratified CV) and regressor (R², MAE) on circuit, weather, field-size, and season features—with position-change aggregates omitted from inputs so labels are not leaked.

## Limitations

- "Position change" is a proxy for overtaking; it does not capture mid-race moves that are reversed by the finish.
- Weather data comes from the live-timing feed and may miss brief localised showers.
- External factors (safety cars, red flags, race-control decisions) are not modelled.
