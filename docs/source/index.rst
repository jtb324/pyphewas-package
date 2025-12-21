.. PyPheWAS documentation master file, created by
   sphinx-quickstart on Tue Dec  9 10:14:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPheWAS documentation
======================

**PyPheWAS** is my personal Python script for running PheWAS within Python. It is highly influenced by both the R `PheWAS package <https://github.com/PheWAS/PheWAS/>`_ and the Python `PheTK package <https://github.com/nhgritctran/PheTK>`_ package while *hopefully* providing some beneficial differences. This package allows users to run either a linear or logistic regression model for the PheWAS and uses modern Python libraries such as "polars" to bring better performance in larger datasets.


Installation:
-------------
This code is hosted on PYPI and can be installed using Pip. It is recommended to install the package into a virtual environment. The PyPheWAS-package only supports Python 3.11.1+ so make sure that you have this installed on your computer. The commands to do this are shown below

.. code:: python

  #Pip installation
  python3 -m venv pyphewas-venv

  source pyphewas-venv/bin/activate

  pip install pyphewas-package


Using the PyPheWAS package:
---------------------------
The PyPheWAS package has 2 main functions explained below:

* **pyphewas** - This command calls the script that performs the PheWAS for the dataset. The program will determine cases/controls/exclusions from the provided phecode counts file and will use these classifications in the regression. This code will generate p-values, betas, and standard errors for every term in the model (Ex. If you had three predictors age, sex, and record length, then you would have three columns in the output for each predictor representing p-values, betas, and standard errors).

.. note::

  **Definition of case/control/exclusion status for each phecode:**

  Case/control/exclusion definition for each phecodes is performed using the following 3 steps:
  
  1. If an individual has the phecode in their record on 2+ unique dates then they are a case

  2. If the individual has only 1 occurrence of the code in their record then they are excluded (This step is skipped if the user has lowered the minimum case threshold from 2 to 1)

  3. All other individuals are classified as controls


* **make_manhattan** - This command calls the script responsible for generating a Manhattan plot from the PheWAS results. It is designed for the results from the pyphewas command, but has flexibility for other results from other programs as long as that program has p-values and betas.


Example Commands:
-----------------
**Running a PheWAS for binary covariates**

The following command illustrates how to run a PheWAS for the provided test data. In this example, our data represents a binary phenotype of interest where individuals are either cases or controls. For this example we can assume that all test data is in the "tests/inputs" directory of the repository (You can go to the repository using the github icon at the top right section of the page near the contents menu). This command filters the phecode counts so that cases must have 2 or more occurrences of the phecode. Additionally, this example only includes phecodes which have 100 or more cases.

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

To run a linear regression it is as simple as changing the model flag from "logistic" to "linear" as shown below. This will change the model from statsmodel.formula logit to the statsmodel.formula glm with the Gaussian family.

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

**Generating a Manhattan plot from the data**

The PyPheWAS package also has the ability to generate a Manhattan plot from the output of the pyphewas command. The command below uses the output from the example pyphewas commands above to generate a (*bad in this case*) Manhattan plot. The user is expected to provide the name of the columns with the p-values and betas for the variable of interest.

.. code:: bash

  make_manhattan \
    --input-file output.txt.gz \
    --output-file output_test_manhattan \
    --pval-col predictor_pvalue \
    --beta-col predictor_beta 


.. note::

  This functionality can also be used in a jupyter notebook. The different python functions used in the make_manhattan command are exposed through the package. This example only shows how to run the CLI command, but users can read more about running the code in a jupyter notebook in the "insert section here".

    

.. toctree::
   :maxdepth: 2
   :hidden:

   inputs_and_outputs/index

