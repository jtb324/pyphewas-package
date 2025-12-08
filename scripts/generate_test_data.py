import pandas as pd
import numpy as np


def generate_phewas_data(n_subjects=500, n_phecodes=50, signal_strength=2.0):
    """
    Generates a synthetic PheWAS dataset.

    Args:
        n_subjects: Number of rows (keep <1000 for fast CI)
        n_phecodes: Number of phenotype columns
        signal_strength: Odds ratio multiplier for the 'true positive'
    """
    np.random.seed(42)  # Fixed seed for reproducible CI

    # 1. Generate IDs and Covariates
    ids = [f"ID_{i}" for i in range(n_subjects)]
    age = np.random.normal(50, 10, n_subjects)
    sex = np.random.randint(0, 2, n_subjects)  # 0=F, 1=M

    # 2. Generate Genotype (SNP)
    # 0, 1, 2 based on Hardy-Weinberg principle (Minor Allele Freq = 0.3)
    genotype = np.random.choice([0, 1, 2], size=n_subjects, p=[0.49, 0.42, 0.09])

    data = pd.DataFrame({"id": ids, "genotype": genotype, "age": age, "sex": sex})

    # 3. Generate Phecodes (Binary Phenotypes)
    # Most phecodes are sparse (mostly zeros) and uncorrelated
    phecode_cols = []

    for i in range(n_phecodes):
        code_name = f"phecode_{i}"

        # Base probability of having the condition (prevalence)
        base_prob = np.random.uniform(0.05, 0.2)

        # Calculate Log-odds
        # log(p / 1-p) = beta0 + beta_geno*G + beta_age*Age
        logit = np.log(base_prob / (1 - base_prob))

        # --- INJECT SIGNAL ---
        # We make 'phecode_10' highly correlated with the genotype
        if i == 10:
            logit += signal_strength * genotype
        else:
            # Add random noise for null variables
            logit += 0.01 * np.random.randn(n_subjects)

        # Convert back to probability
        prob = 1 / (1 + np.exp(-logit))

        # Bernouilli trial to determine status (0 or 1)
        status = np.random.binomial(1, prob)
        data[code_name] = status
        phecode_cols.append(code_name)

    return data, phecode_cols


if __name__ == "__main__":
    df, codes = generate_phewas_data()
    # Save to the tests folder
    df.to_csv("tests/synthetic_phewas_data.csv", index=False)
    print(f"Generated data with {len(df)} subjects and {len(codes)} phecodes.")
    print("True positive signal injected at: phecode_10")
