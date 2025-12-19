#!/bin/bash
# Bash script that can be used to generate the test data
RELATIVE_TEST_INPUTS_DIR=../tests/inputs/
PYTHON_SCRIPT=./generate_test_data.py

# Making sure the test inputs directory exists
if [ ! -d $RELATIVE_TEST_INPUTS_DIR ]; then
  echo "Creating the directory for test data: ${RELATIVE_TEST_INPUTS_DIR}"
  mkdir $RELATIVE_TEST_INPUTS_DIR
else
  echo "The directory, ${RELATIVE_TEST_INPUTS_DIR}, already exists"
fi

echo "generating binary phenotype data"

python3 $PYTHON_SCRIPT \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type binary \
  --output-dir $RELATIVE_TEST_INPUTS_DIR \
  --target-phecode-index 10 \
  --target-phecode-count 80 \
  --signal-phecode-index 15 \
  --seed 1234 \
  --save-plots \
  --log-filename test_binary_predictory_with_min_phecode.log

echo "generating continuous phenotype data"

python3 $PYTHON_SCRIPT \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type dosage \
  --output-dir $RELATIVE_TEST_INPUTS_DIR \
  --target-phecode-index 10 \
  --target-phecode-count 80 \
  --signal-phecode-index 15 \
  --seed 1234 \
  --save-plots \
  --log-filename test_dosage_with_min_phecode.log

echo "generating binary phenotype with an example of perfect separation"

# This command will ensure that there is perfect separation for a phenotype to test firth regression
python3 $PYTHON_SCRIPT \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type binary \
  --output-dir $RELATIVE_TEST_INPUTS_DIR \
  --seed 1234 \
  --inject-perfect-separation \
  --perfect-separation-index 10 \
  --signal-phecode-index 15 \
  --save-plots \
  --log-filename test_perfect_separation.log

echo finished creating the test inputs
