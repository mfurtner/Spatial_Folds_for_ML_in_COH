# Spatial_Folds_for_ML_in_COH
Python functions and scripts to generate spatial folds for cross-validation in ML models via assigned user-designated spatial clusters of cave and sinkhole sites in the Cradle of Humankind, South Africa.

#### Contains:
TrainingObservations_Positive.csv : a .csv file containing all positive training observations and input variables. Includes a 'Label' column to designate site or site type, 'Target' column to designate observation type, 'Cluster' column which reflects user-assigned cluster numbers based on the close spatial proximity to other related training observations, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset does not include coordinate information for privacy reasons.

TrainingObservations_Negative.csv : a .csv file containing all negative training observations and input variables. Includes a 'Label' column to designate site type, 'Target' column to designate observation type, 'Cluster' column which to match positive dataset although the negative training observations are not spatially autocorrelated, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset also does not include coordinate information.

create_spatial_folds.py : python functions and scripts to automatically generate a reproducible set of training and test folds for cross-validation in ML models

train_folds.pkl : pickle object containing training folds output, readable as a list of 10 dataframes

test_folds.pkl : pickle object containing test folds output, readable as a list of 10 dataframes
