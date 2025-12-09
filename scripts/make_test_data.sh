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

python3 $PYTHON_SCRIPT \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type binary \
  --output-dir $RELATIVE_TEST_INPUTS_DIR \
  --target-phecode-index 10 \
  --min-phecode-count 80 \
  --seed 1234

python3 $PYTHON_SCRIPT \
  --sample-count 10000 \
  --phecode-count 30 \
  --predictor-type dosage \
  --output-dir $RELATIVE_TEST_INPUTS_DIR \
  --target-phecode-index 10 \
  --min-phecode-count 80 \
  --seed 1234

echo finished creating the test inputs
