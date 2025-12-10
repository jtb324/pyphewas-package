Manhattan Inputs and Outputs
============================
To visualize results from the PheWAS, the PyPheWAS-package has a command called "make_manhattan" that allows uses to generate a manhattan plot of the results.





Output Plot:
------------
Noteable features of the output plot are described below:

* Different phecode categories are illustrated by different colors on the x-axis

* bonferroni significance threshold is indicated by the red dashed line

* infinity line is indicated by the dashed blue lines. In some PheWASes the results can be so significant that they get rounded to zero. To account for this, the PyPheWAS-package finds the most significant, no rounded result and multiplies by 1.02 to generate the infinity line.
