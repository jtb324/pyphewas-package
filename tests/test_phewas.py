from pathlib import Path
from pyphewas.run_PheWAS import read_in_cases_and_exclusions

TESTS_DIR = Path(__file__).parent
INPUTS_DIR = TESTS_DIR / "inputs"
COUNTS_FILE = (
    INPUTS_DIR / "phecodes_file_10000_samples_binary_predictor_30_phecodes.txt"
)
PHECODE_COUNT_IN_FILE = 30


def test_read_in_cases() -> None:
    """Read in phecode counts and check if a phecode with all cases is being generated correctly"""
    # Test with min_phecode_count = 2
    phecode_map = read_in_cases_and_exclusions(COUNTS_FILE, min_phecode_count=2)

    # phecode_0: 620 entries, all >= 2. Should be 620 cases, 0 exclusions.
    assert "phecode_0" in phecode_map
    assert len(phecode_map["phecode_0"].cases) == 620
    assert len(phecode_map["phecode_0"].exclusions) == 0


def test_read_in_exclusions() -> None:
    """Make sure the code is also handling the case where the phecode has exclusions"""
    # phecode_10: 80 entries, all == 1. Should be 0 cases, 80 exclusions.
    phecode_map = read_in_cases_and_exclusions(COUNTS_FILE, min_phecode_count=2)

    assert "phecode_10" in phecode_map
    assert len(phecode_map["phecode_10"].cases) == 0
    assert len(phecode_map["phecode_10"].exclusions) == 80


def test_all_phecodes_read_in() -> None:
    """Make sure that the python dictionary has 30 keys meaning all of the
    phecodes were read in"""
    phecode_map = read_in_cases_and_exclusions(COUNTS_FILE, min_phecode_count=2)

    assert (
        len(phecode_map.keys()) == PHECODE_COUNT_IN_FILE
    ), f"Expected the program to read in {PHECODE_COUNT_IN_FILE} from the file {COUNTS_FILE}. Instead {len(phecode_map.keys())} phecodes were read in"
