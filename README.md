# Multi-Species, Multi-Reaction Whole Cell Adaptation Manuscript
----------------------------------------------------------------
This whole cell adaption of the thermodynamic framework of the Multi-Species, Multi-Reaction model is aimed at providing insight with physics-based models on fundamental electrode information in a noninvasive, nondestructive manner through analyzing open-circuit voltage and differential voltage data. This repository contains all the necessary code and data used in the corresponding manuscript. This repository can be cited with: 

For further information or if this code is used, please go to or cite the following paper:

* Paper citation will be available upon acceptance *

----------------------------------------------------------------
### Software Dependencies
----------------------------------------------------------------
This repository was developed using the following versions of the subsequent softwares:

* Python 3.7.6
* Conda 4.6.14
* Git Bash for Windows

The conda environment used for this work can be recreated with the following commands:

```conda env create -f environment.yml```

```conda activate msmr```

----------------------------------------------------------------
### Folders
----------------------------------------------------------------

*jupyter*: This folder contains two Jupyter Notebooks (Supplementary Notebook and User Guide) and the necessary utilities
needed for importing, analyzing, and compiling the data into the manuscruipt and supplementary figures. The User Guide
provides a short tutorial on how to use the software to fit differential voltage using the MSMR model.

*supplementary-files*: This folder contains the experimental open-circuit voltage data and the summary of the bootstrapped
data needed for the analysis performed in this manuscript.

*figures*: This folder contains all the figures used in the manuscript and the subsequent Supplementary Information.


----------------------------------------------------------------




