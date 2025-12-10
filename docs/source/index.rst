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


Using the PyPheWAS package:
---------------------------
The PyPheWAS package has 2 main functions explained below:

* **pyphewas** - This command calls the script that performs the PheWAS for the dataset. This code will generate p-values, betas, and standard errors for every term in the model (Ex. If you had three predictors age, sex, and record length, then you would have three columns in the output for each predictor representing p-values, betas, and standard errors.)

* **make_manhattan** - This command calls the script responsible for generating a manhattan plot from the PheWAS results. It is designed for the results from the pyphewas command, but has flexibility for other results from other programs as long as that program has p-values and betas.


Example Commands:
-----------------
**Running a PheWAS for binary covariates**

The following command illustrates how to run a phewas for the provided test data. In this example, our data represents a binary phenotype of interest where individuals are either cases or controls. For this example we can assume that all test data is in the "tests/inputs" directory of the repository. This command filters the phecode counts so that cases must have 2 or more occurences of the phecode. Additionally, this example only includes phecodes which have 100 or more cases. . 

.. code:: bash

  pyphewas \
      --counts phecodes_file_10000_samples_binary_predictor_30_phecodes.txt \
      --covariate-file covariates_file_10000_samples_binary_predictor_30_phecodes.txt \
      --min-phecode-count 2 \
      --status-col predictor \
      --sample-col id \
      --covariate-list age sex \
      --min-case-count 100 \
      --cpus 2 \
      --output output.txt.gz \
      --phecode-version None \
      --model logistic 

To run a linear regression it is as simple as changing the model flag from "logistic" to "linear" as shown below. This will change the model from statsmodel.formula logit to the statsmodel.formula glm with the Gaussian family
. 

.. code:: bash


  pyphewas \
      --counts phecodes_file_10000_samples_dosage_predictor_30_phecodes.txt \
      --covariate-file covariates_file_10000_samples_dosage_predictor_30_phecodes.txt \
      --min-phecode-count 2 \
      --status-col predictor \
      --sample-col id \
      --covariate-list age sex \
      --min-case-count 100 \
      --cpus 2 \
      --output output.txt.gz \
      --phecode-version None \
      --model linear

**Generating a manhattan plot from the data**
The PyPheWAS package also has the ability to generate a manhattan plot from the output of the pyphewas command. This functionality can also be used in a juypter notebook. The different python functions used in the make_manhattan command are exposed through the package. This example only shows how to run the CLI command, but users can read more about running the code in a jupyter notebook in the "insert section here".

.. code:: bash

  make_manhattan \
    

.. toctree::
   :maxdepth: 2
   :hidden:

   pyphewas

