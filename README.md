# Affinity_Gaus-MC
This repository has the latest updates with all the modifications that have been made for both Gaussian and MC generated data


<img width="895" alt="Screenshot 2025-04-15 at 23 23 20" src="https://github.com/user-attachments/assets/8a72051d-6221-4235-827e-c506a453fdf0" />


Everything you see on a picture above represents the update until April 15, 2025.  
Here is descriptions of files you see:

* Folder "generated_data" contains Kinematical (and some of the files contain non-perturbative parameters) variables that are needed to run the   "driver".
    * "CLAS12_1400.xlsx" - contains kinematical variables
    * "CLAS12_Mki_Mkf_deltakt_kit_w_all_1400.xlsx" - contains not only kinematical variables but also non-perturbative parameters + angles for 1,400 events
    * "CLAS12_all_values_1k.csv" - contains not only kinematical variables but also non-perturbative parameters + angles for 1,000 events
    * "jlab12_MC_data.xlsx" - binned data from Rowan
    * "jlab12.xlsx" - binned data from Harut

* "All_Parameters.ipynb" - code for making plots for parameters
* "Finding similar bins.ipynb" - code that finds similar bins in 2 different datasets
    * "Harut_similar_bin.xlsx", "Rowan_similar_bin.xlsx" - result of the code
* "JLab12-Harut_data.ipynb" - code for plotting *Affinity* for dataset from Harut ("jlab12.xlsx")
* "JLab12-Rowan_data.ipynb" - code for plotting *Affinity* for dataset from Rowan ("jlab12_MC_data.xlsx")
* "Kinematical range plots.ipynb" - code for plotting kinematical ranges
* "Region Indicators MC VS Gaussian.ipynb" - code for plotting region indicators
* "Region Indicators MC VS Gaussian + kinematics.ipynb" - code for plotting region indicators + kinematics
* "driver_Gaussian_plus_filter.ipynb" - code for *affinity tool* with the condition $k_{it}^2 - M_{ki}^2$
* "driver_Rowan_Non_pert.ipynb" - code for *affinity tool* with all the parameters (non-perturbative, partonic) taken from MC generation
* "driver_Rowan_Alexey_suggestions.ipynb" - code for *affinity tool* with parameters $M_{ki}$, $k_{it}$ and $\delta k_t$ taken from MC generation
* "driver_old_version.ipynb" - code for *affinity tool* with all the parameters are generated with Gaussian distribution  

## **Please note, that all these files do not contain any changes which were suggested at the meeting on 14th of April 2025**
