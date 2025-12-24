import pandas as pd
import numpy as np
from collections import Counter

RAW_DATA_PATH = "data/raw/HouseDataRaw.csv"

HIGH_CARD_FEATURE = "address"
CAT_FEATURE = "district"

PSI_THRESHOLD = 0.25


def clean_raw_columns(df):
    df[HIGH_CARD_FEATURE] = df[HIGH_CARD_FEATURE].astype(str)
    df[CAT_FEATURE] = df[CAT_FEATURE].astype(str)
    return df


def calculate_psi(expected, actual, eps=1e-6):
    psi = 0.0
    for e, a in zip(expected, actual):
        psi += (a - e) * np.log((a + eps) / (e + eps))
    return psi


def categorical_psi(ref_series, prod_series):
    ref_counts = Counter(ref_series)
    prod_counts = Counter(prod_series)

    all_keys = set(ref_counts) | set(prod_counts)

    ref_dist = []
    prod_dist = []

    for k in all_keys:
        ref_dist.append(ref_counts.get(k, 0))
        prod_dist.append(prod_counts.get(k, 0))

    ref_dist = np.array(ref_dist) / sum(ref_dist)
    prod_dist = np.array(prod_dist) / sum(prod_dist)

    return calculate_psi(ref_dist, prod_dist)


def cardinality_ratio(ref_series, prod_series):
    return prod_series.nunique() / max(ref_series.nunique(), 1)


def monitor_drift(reference_df, production_df):
    print("Running drift monitoring...\n")

    district_psi = categorical_psi(
        reference_df[CAT_FEATURE],
        production_df[CAT_FEATURE]
    )

    print(f"District PSI: {district_psi:.4f}")
    if district_psi > PSI_THRESHOLD:
        print("️ALERT: District distribution drift detected")

    addr_ratio = cardinality_ratio(
        reference_df[HIGH_CARD_FEATURE],
        production_df[HIGH_CARD_FEATURE]
    )

    print(f"Address cardinality ratio: {addr_ratio:.2f}")
    if addr_ratio > 1.3:
        print("️ALERT: Address cardinality drift detected")

    print("\nMonitoring complete.")


def simulate_production_data(reference_df, frac=0.3):
    prod_df = reference_df.sample(frac=frac, random_state=42).copy()

    # simulate drift
    prod_df[CAT_FEATURE] = prod_df[CAT_FEATURE].sample(frac=1.0).values
    prod_df[HIGH_CARD_FEATURE] = prod_df[HIGH_CARD_FEATURE] + "_new"

    return prod_df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_raw_columns(df)

    ref_df = df.sample(frac=0.7, random_state=42)
    prod_df = simulate_production_data(ref_df)

    monitor_drift(ref_df, prod_df)
