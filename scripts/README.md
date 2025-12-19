# Synthetic PheWAS Data Generator

This directory contains `generate_test_data.py`, a utility script designed to create synthetic datasets for testing the `pyphewas` package. It generates a covariate file (demographics + predictor) and a phecode counts file (phenotypes).

The script allows for fine-grained control over the data generation process, including:
- Setting the sample size and number of phecodes.
- Choosing between binary (case/control) or continuous (dosage) predictors.
- Injecting statistical signals into specific phecodes.
- Creating scenarios with "perfect separation" to test robust statistical methods (e.g., Firth regression).
- Controlling phecode prevalence to test filtering logic.
- Visualizing demographic distributions (Age/Sex).

## Usage

```bash
python generate_test_data.py [OPTIONS]
```

## Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--sample-count` | `int` | `1000` | Number of subjects (rows) to generate. |
| `--phecode-count` | `int` | `50` | Number of phenotype columns to generate. |
| `--signal-strength` | `float` | `1.5` | The coefficient (log-odds) for the injected signal. |
| `--predictor-type` | `str` | `binary` | Type of predictor: `'binary'` (case/control) or `'dosage'` (continuous 0-2). |
| `--output-dir` | `path` | `./synthetic_phewas_data` | Directory to save the output files and plots. |
| `--mean-age` | `int` | `50` | Mean age for the generated normal distribution (SD=15). |
| `--min-phecode-prevalence` | `float` | `0.05` | Minimum background probability for random noise variables. |
| `--target-phecode-index` | `int` | `0` | Index of the phecode to apply a specific count to (used with `--target-phecode-count`). |
| `--target-phecode-count` | `int` | `0` | Forces a specific number of cases for the phecode at `target-phecode-index`. Overrides min prevalence. |
| `--signal-phecode-index` | `int` | `10` | Index of the phecode to inject a significant relationship with the predictor. |
| `--inject-perfect-separation` | `flag` | `False` | If set, injects perfect separation logic. |
| `--perfect-separation-index` | `int` | `-1` | Index of the phecode to have perfect separation (requires flag above). |
| `--log-filename` | `str` | **Required** | Filename for the run log. |
| `--seed` | `int` | `1234` | Random seed for reproducibility. |
| `--save-plots` | `flag` | `False` | If set, saves demographic plots to the output directory. |

## Feature Guides

### 1. Predictor Types
You can generate data for two types of studies:
*   **Binary:** Simulates a Case/Control study (e.g., Disease vs Healthy).
    *   `--predictor-type binary`
*   **Dosage:** Simulates a genetic study or continuous variable (e.g., values 0.0 - 2.0).
    *   `--predictor-type dosage`

### 2. Controlling Signals
To test if your association analysis finds true positives, you can inject a signal:
*   Use `--signal-phecode-index 15` to choose which column has the signal.
*   Use `--signal-strength 1.5` to define how strong the association is (log-odds).

### 3. Perfect Separation
"Perfect separation" occurs when a predictor perfectly predicts the outcome (e.g., all cases have the mutation, no controls do). Standard logistic regression fails here.
*   Add `--inject-perfect-separation` to enable this mode.
*   Use `--perfect-separation-index` to specify which phecode gets this pattern.

### 4. Minimum Counts & Filtering
To test how your pipeline handles rare codes:
*   **General Noise:** Use `--min-phecode-prevalence` to set a baseline rarity for all "noise" phecodes.
*   **Specific Control:** Use `--target-phecode-count 80` combined with `--target-phecode-index 10` to ensure a specific phecode has exactly 80 cases (useful for testing "min. case count" thresholds).

### 5. Demographic Plots
If the `--save-plots` flag is provided, the script generates histograms for Age and Sex distributions in a `plots/` subdirectory within your output folder. The filenames are suffixed with run parameters to avoid overwriting.

## Examples

These examples are adapted from `make_test_data.sh`.

### Generate Binary Phenotype Data
Creates a dataset with 10,000 samples and 30 phecodes. It ensures phecode 10 has exactly 80 cases and injects a signal into phecode 15.

```bash
python generate_test_data.py \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type binary \
  --output-dir ../tests/inputs/ \
  --target-phecode-index 10 \
  --target-phecode-count 80 \
  --signal-phecode-index 15 \
  --seed 1234 \
  --log-filename test_binary_predictory_with_min_phecode.log
```

### Generate Continuous (Dosage) Phenotype Data
Similar to above, but the predictor is a continuous "dosage" value (0-2).

```bash
python generate_test_data.py \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type dosage \
  --output-dir ../tests/inputs/ \
  --target-phecode-index 10 \
  --target-phecode-count 80 \
  --signal-phecode-index 15 \
  --seed 1234 \
  --log-filename test_dosage_with_min_phecode.log
```

### Generate Data with Perfect Separation
Creates a dataset specifically to test Firth regression or other penalized methods. Phecode 10 will have perfect separation with the predictor.

```bash
python generate_test_data.py \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type binary \
  --output-dir ../tests/inputs/ \
  --seed 1234 \
  --inject-perfect-separation \
  --perfect-separation-index 10 \
  --signal-phecode-index 15 \
  --log-filename test_perfect_separation.log
```

