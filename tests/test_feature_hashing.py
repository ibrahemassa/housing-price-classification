import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.training.preprocess import build_preprocessor


def test_feature_hasher_is_deterministic():
    _, hasher = build_preprocessor()

    sample = [["Kadikoy", "NaturalGas", "3rdFloor"]]

    hash_1 = hasher.transform(sample).toarray()
    hash_2 = hasher.transform(sample).toarray()

    assert np.array_equal(hash_1, hash_2)


def test_feature_hasher_diff_input_diff_output():
    _, hasher = build_preprocessor()

    input_a = [["Kadikoy", "NaturalGas", "3rdFloor"]]
    input_b = [["Besiktas", "Electric", "1stFloor"]]

    hash_a = hasher.transform(input_a).toarray()
    hash_b = hasher.transform(input_b).toarray()

    assert not np.array_equal(hash_a, hash_b)


def test_feature_hasher_output_shape():
    _, hasher = build_preprocessor()

    sample = [["address_1", "2023-01", "unknown"]]

    hashed = hasher.transform(sample)

    assert hashed.shape == (1, 2**14)
