Inputs and Outputs
==================

This document describes all of the input flags for the pyphewas command as well as the different columns in the output file


PheWAS Inputs:
---------------
**Required Inputs**

* **--counts**: filepath to a comma separated file where each row has a ID, a phecode id, and the number of times that individual has that phecode in their medical record. The file should only have 3 columns and a error will be raised if it has any other amounts of columns.

.. list-table:: Example counts input structure
   :widths: 25 25 50
   :header-rows: 1

   * - person_id
     - phecode_id
     - count
   * - 1001
     - 250.2
     - 5
   * - 1002
     - 401.1
     - 12
   * - 1003
     - 296.22
     - 1

* **--covariate-file**: filepath to a comma separated file that list the covariates and predictor for each individual. This column containing sample ids should be named the same as the corresponding column in the counts file. The individuals listed in the covariate file will be the individuals in the cohort. *Note* If the 'flip-predictor-and-outcome' flag is used then the predictor variable is assumed to be the outcome in the model.

* **--covariate-list**: Space separated list of covariates to use in the model. All of these covariates must be present in the covariate file and must be spelled exactly the same otherwise the code will crash.

* **--phecode-version**: String telling which version of phecodes to use. This argument helps with mapping the PheCode ID to a description. The allowed values are "phecodeX", "phecode1.2", and "phecodeX_who". Most users will only need to use either the PhecodeX or Phecode1.2 option.

**Optional Inputs**

* **--min-phecode-count**: Minimum number of phecodes an individual is required to have in order to be considered a case for a phecode. Default value is 2. Under default settings, all individuals with 1 occurence of the phecode are excluded from the regression. If this value is set to 1 then there are no excluded individuals.

* **--min-case-count**: Minimum number of cases a phecode has to have to be included in the analysis. The default value is 20. There is no rigorous testing behind this value, only convention. For more rigorous results, a more conservative value of 100 may be ideal.

* **--status-col**: Column name for the column in the covariate file that has the predictor case/control status. This file should be a comma-separated file. Default value is "status"

* **--sample-col**: Column name for the column in the covariates file that has the individual ids. Default value is "person_id"

* **--output**: Filename to write the output to. The output will be written as a tab separated file. If the suffix of the file ends in gz then the file will be gzipped otherwise the file will be uncompressed. Default value is test_output.txt

* **--phecode-descriptions**: filepath to a comma separated file that list the phecode ID and the corresponding phecode name. There are default description files stored in the './src/phecode_maps/' folder if you wish to see example files that are currently used in the code. The phecode ID is expected to be the first column while the phecode description is expected to be the 4th column.

* **--model**: Type of regression model to use for the analysis. The two options are 'logistic' or 'linear'. Default option is logistic.

* **--cpus**: Number of cpus to use during the analysis. Default value is 1.

* **--max-iterations**: Number of iterations for the regression to try to converge. If the model doesn't converge after reaching the max iteration threshold then a ConvergenceWarning will be thrown. If you run this code and find that many PheCodes are not converging then it is recommended to increase this value to attempt to get more phecodes to converge. Default value is 200

* **--flip-predictor-and-outcome**: Depending on the analysis, you may want the status column in the covariate file to be a predictor or to be the outcome. If you want the status to be the outcome then you can supply this flag as '--flip-predictor-and-outcome'. When the status is the outcome, then the case/control status for the individual phecodes will become the predictor.

* **--run-sex-specific**: Depending on the analysis, you may also want to restrict the analysis to a sex stratified cohort. This command is one of three flags that have to be used in tandem that allow you to stratify the analysis. Allowed values are 'male-only' and 'female-only'.

* **--male-as-one**: If the '--run-sex-specific' flag is used then this flag also has to be passed indicating if males were coded as 1 and females as 0 or vice verse. You could pass this flag as '--male-as-one' to indicate that males were coded as 1. The default value is True although this flag will be ignored if the '--run-sex-specific' flag is not provided.

* **--sex-col**: Column name of the column in the covariate field containing Sex or Gender information. This flag is required if the '--run-sex-specific' flag was used. 

* **--phecode-descriptions**: Comma separated file that list phecode ids and then the name description. The phecode ID is the first column and then the fourth column should be the phecode description. This information is used to give names for the Phecode ids used in the analysis for easier interpretation. (Ex: 1st column: 499, 4th column: "Cystic Fibrosis").




PheWAS Output:
--------------
