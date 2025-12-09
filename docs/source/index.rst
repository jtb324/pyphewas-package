.. PyPheWAS documentation master file, created by
   sphinx-quickstart on Tue Dec  9 10:14:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPheWAS documentation
======================

**PyPheWAS** is my personal python script for running PheWAS within python. It is highly influenced by both the R `PheWAS package <https://github.com/PheWAS/PheWAS/>`_ and the python `PheTK package <https://github.com/nhgritctran/PheTK>`_ package while *hopefully* providing some beneficial differences. This package allows users to run either a linear or logistic regression model for the phewas and uses modern python libraries such as "polars" to bring better performance in larger dataset.


Installation:
-------------
This code is hosted on PYPI and can be installed using Pip. It is recommended to install the package into a virtualenv. The commands to do this are shown below

.. code:: python

  #Pip installation
  python3 -m venv pyphewas-venv

  source pyphewas-venv/bin/activate

  pip install pyphewas-package


Example Commands:
-----------------
**Non sex stratified with parallelization**:

.. code:: bash

  pyphewas \
      --counts counts.csv \
      --covariate-file covariates.csv \
      --min-phecode-count 2 \
      --status-col status \
      --sample-col person_id \
      --covariate-list EHR_GENDER age unique_phecode_count \
      --min-case-count 100 \
      --cpus 25 \
      --output output.txt.gz \
      --phecode-version phecodeX

**Sex Stratified with parallelization**:

.. code:: bash

  pyphewas \
      --counts counts.csv \
      --covariate-file covariates.csv \
      --min-phecode-count 2 \
      --status-col status \
      --sample-col person_id \
      --covariate-list age unique_phecode_count \
      --min-case-count 100 \
      --cpus 25 \
      --output output.txt.gz \
      --phecode-version phecodeX \
      --flip-predictor-and-outcome \
      --run-sex-specific female-only \
      --male-as-one True \
      --sex-col EHR_GENDER

.. toctree::
   :maxdepth: 2
   :caption: Contents:

