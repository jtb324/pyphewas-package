import pandas as pd
from pathlib import Path
import numpy as np
import numpy.typing as npt
import argparse
from rich_argparse import RichHelpFormatter


def generate_covariates(
    n_subjects: int, predictor_type: str, seed: int = 42, mean_age: int = 50
) -> pd.DataFrame:
    """
    Generates a DataFrame containing IDs, Age, Sex, and the Predictor variable.

    Parameters
    ----------
    n_subjects : int
        number of participants to generate covariate data for

    predictor_type : str
        whether to generate data for a binary phenotype fo
        interest or a continuous one

    seed : int
        random seed to make sure that the rng stays the same
        between different runs

    mean_age : int
        mean age to be used in the covariates. A normal distribution will be generated will this value as the mean
    """
    # Local seed for this function to ensure stability
    rng = np.random.default_rng(seed)

    ids = [f"ID_{i}" for i in range(n_subjects)]

    # Age: Normal dist, mean 50, sd 15
    age = rng.normal(mean_age, 15, n_subjects)

    # Sex: 0 or 1
    sex = rng.integers(0, 2, n_subjects)

    # Predictor Logic
    if predictor_type == "binary":
        # Case/Control (0 or 1)
        predictor_data = rng.binomial(1, 0.5, size=n_subjects)
        col_name = "predictor"

    elif predictor_type == "dosage":
        # Continuous 0.0 to 2.0
        raw_geno = rng.choice([0, 1, 2], size=n_subjects, p=[0.49, 0.42, 0.09])
        noise = rng.normal(0, 0.15, size=n_subjects)
        predictor_data = np.clip(raw_geno + noise, 0.0, 2.0)
        predictor_data = np.round(predictor_data, 3)
        col_name = "predictor"
    else:
        raise ValueError("Invalid predictor_type")

    df = pd.DataFrame({"id": ids, col_name: predictor_data, "age": age, "sex": sex})

    return df


def generate_phecode_counts(
    participant_list: list[str],
    n_subjects: int,
    n_phecodes: int,
    signal_strength: float,
    min_percent: float,
    predictor_data: npt.NDArray,
    seed: int = 42,
    target_phecode_count: int = 0,
    target_phecode_index: int = 0,
) -> pd.DataFrame:
    """
    Generates a Long-Format DataFrame of phecode counts.
    Requires the covariate_df to inject signal based on
    the predictor column.

    Parameters
    ----------
    covariate

    Raises
    ------
    AssertionError
        raises an assertation error if the number of participants
        to create phecode data for is different than the number of
        people that we have case/control statuses for.
    """
    assert len(participant_list) == len(
        predictor_data
    ), f"The number of participants to generate data (n={len(participant_list)})for is not the same as the number of individuals in the case/control status list (n={len(predictor_data)})"

    assert (
        target_phecode_count <= n_subjects
    ), f" Expected the target phecode count to be less than the number of samples. Instead requested the phecode count to be {target_phecode_count} when the number of samples was {n_subjects}"

    rng = np.random.default_rng(seed + 1)  # Offset seed slightly

    # Prepare logic for probabilities
    max_prob = 0.2  # Max is 20%. This would be a very common phenotype

    # Temp dictionary to store columns before melting
    # We start with ID so we can merge later
    data_dict = {"id": participant_list}
    phecode_cols = []

    for i in range(n_phecodes):
        code_name = f"phecode_{i}"
        phecode_cols.append(code_name)
        if i == target_phecode_index and target_phecode_count > 0:
            # Create an array of zeros for all subjects
            final_counts = np.zeros(n_subjects, dtype=int)

            # Randomly select subjects to have this phecode
            selected_indices = rng.choice(
                n_subjects, size=target_phecode_count, replace=False
            )

            # Set their count to 1
            final_counts[selected_indices] = 1

        else:
            # 1. Determine Baseline Probability
            if i == 10:
                # Target signal is always somewhat common so we can detect it
                target_base = max(0.05, min_percent)
                base_prob = rng.uniform(target_base, max_prob)
            else:
                # Noise variables (Log-Uniform distribution)
                log_prob = rng.uniform(np.log(min_percent), np.log(max_prob))
                base_prob = np.exp(log_prob)

            # 2. Linear Predictor (Logit)
            intercept = np.log(base_prob / (1 - base_prob))
            logit = intercept + (0.05 * rng.standard_normal(n_subjects))

            # 3. Inject Signal
            if i == 10:
                logit += signal_strength * predictor_data

            # 4. Convert to Binary Presence (0 or 1)
            probs = 1 / (1 + np.exp(-logit))
            has_disease = rng.binomial(1, probs)

            # 5. Generate Counts (Geometric distribution for count > 0)
            # 1 + Geometric(p=0.4) creates a "billing code" distribution (1, 2, 3...)
            random_counts = 1 + rng.geometric(p=0.4, size=n_subjects)

            # Apply mask: if has_disease is 0, count is 0
            final_counts = has_disease * random_counts

        data_dict[code_name] = final_counts

    # Create wide DataFrame
    wide_df = pd.DataFrame(data_dict)

    # Melt to Long Format
    long_df = wide_df.melt(
        id_vars=["id"], value_vars=phecode_cols, var_name="phecode", value_name="count"
    )

    # Remove rows where count is 0 (sparse format)
    final_df = long_df[long_df["count"] > 0].reset_index(drop=True)

    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic PheWAS data with injected signals.",
        formatter_class=RichHelpFormatter,
    )

    # Arguments
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="Number of subjects (rows) to generate. (default: %(default)s",
    )
    parser.add_argument(
        "--phecode-count",
        type=int,
        default=50,
        help="Number of phenotype columns to generate. (default: %(default)s",
    )
    parser.add_argument(
        "--signal-strength",
        type=float,
        default=1.5,
        help="The coefficient (log-odds) for the injected signal. (default: %(default)s)",
    )
    # UPDATED: Added 'dosage' to choices
    parser.add_argument(
        "--predictor-type",
        type=str,
        default="binary",
        choices=["binary", "dosage"],
        help="Type of predictor: 'dosage' (continuous 0-2), or 'binary' (case/control). (default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./synthetic_phewas_data"),
        help="Directory to write the output files into. (default: %(default)s",
    )

    parser.add_argument(
        "--mean-age",
        type=int,
        default=50,
        help="mean age for covariates. This value will be used to generate a normal distribution with this value as the mean. (default: %(default)s)",
    )

    parser.add_argument(
        "--min-phecode-prevalence",
        type=float,
        default=0.05,
        help="minimum phecode prevalence in the dataset. For unit test dataset you can use a larger values such as 5% so that you avoid internal errors like colinearity. For bigger integration test with more phecodes you can use rarer phenotypes to determine how the program will behave in this situations and ensure that error handling is catching the appropriate expceptions. (default: %(default)s) ",
    )
    parser.add_argument(
        "--target-phecode-index",
        type=int,
        default=0,
        help="The index of the phecode (e.g., 0 for phecode_0, 1 for phecode_1) that should have the target count specified by --target-phecode-count.",
    )
    parser.add_argument(
        "--min-phecode-count",
        type=int,
        help="Minimum number of samples (count) for a phecode. If set, this overrides --min-phecode-prevalence. Calculates prevalence as min_count / sample_count.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # Lets make sure that the output directory exists:
    if not args.output_dir.exists():
        args.output_dir.mkdir()
        print(
            f"Creating the output directory, {args.output_dir}, to write simulated data into"
        )

    covariate_output_path = args.output_dir / "covariate_df.txt"
    phecode_output_path = args.output_dir / "phecode_output.txt"

    cov_df = generate_covariates(
        args.sample_count, args.predictor_type, args.seed, args.mean_age
    )

    cov_df.to_csv(covariate_output_path, sep=",", index=None)

    print(f"Saved the generated covariates file to {covariate_output_path}")

    samples = cov_df.id.tolist()

    predictor_values = cov_df.predictor.values

    phecode_counts_df = generate_phecode_counts(
        participant_list=samples,
        n_subjects=args.sample_count,
        n_phecodes=args.phecode_count,
        signal_strength=args.signal_strength,
        min_percent=args.min_phecode_prevalence,
        predictor_data=predictor_values,
        seed=args.seed,
        target_phecode_count=args.min_phecode_count,
        target_phecode_index=args.target_phecode_index,
    )

    phecode_counts_df.to_csv(phecode_output_path, sep=",", index=None)

    print(f"Saved the generated phecode counts file to {phecode_output_path}")


if __name__ == "__main__":
    main()
