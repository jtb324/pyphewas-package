Manhattan Inputs and Outputs
============================
To visualize results from the PheWAS, the PyPheWAS-package has a command called "make_manhattan" that allows users to generate a Manhattan plot of the results.


Manhattan Inputs:
-----------------
**Input Arguments**

* **-\-input-file**, **-i**: Input file with the results from the PheWAS. This file should be tab separated and have the columns 'phecode_category', 'phecode_description', and 'converged'.

* **-\-output-file**, **-o**: Filepath to output the Manhattan plot to. Default value is "test.png".

* **-\-pval-col**: Name of the column that has the p-values for the variable of interest. Default value is "pval".

* **-\-beta-col**: Name of the column that has betas for the variables of interest. Default value is "beta".

* **-\-dpi**: Quality of the image to output. Default value is 300.


Output Plot:
------------
Notable features of the output plot are described below:

* Different phecode categories are illustrated by different colors on the x-axis

* Bonferroni significance threshold is indicated by the red dashed line

* Infinity line is indicated by the dashed blue lines. In some PheWASes the results can be so significant that they get rounded to zero. To account for this, the PyPheWAS-package finds the most significant non-zero result and multiplies that value by 1.02 to replace all zero values. The infinity line is drawn at this non-zero value to indicate which values were replaced.

* Positive and negative betas are indicated by the direction of the symbols on the plots
