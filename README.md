# DLSTM2
python code to implement Deep Long-Short Term Memory (updated version)

This model and experiments belongs to a paper by the title:

Time series forecasting of petroleum production using deep LSTM recurrent networks
---------------------------------------------------------------------

There are two folders, namely:

1- case_study_1(chinese oil).

2- case_study_2(indian oil).

each folder contains three subfolders, namely:

1- DLSTM

2- DGRU

3-vanilla RNN

----------------------------------------------------------------------------

# First, DLSTM folder 

contains the python code used to evaluate our model, and this folder contains four files, namely:

1) model_selection_static.py -----> The GA implementation to select model hyperparameters in static scenario

2) model_selection_dynamic.py -----> The GA implementation to select model hyperparameters in dynamic scenario

3) oil_static.py ------> test static scenario

4)  oil_dynamic.py ------> test dynamic scenario

-----------------------------------------------------------------------------

# Second, DGRU folder

contains the python code used to evaluate DGRU model, and this folder contains two files, namely:

1) model_selection.py -----> The GA implementation to select model hyperparameters

2) evaluate.py ------> test DGRU model

--------------------------------------------------------------------------------

# Third, vanilla RNN folder

contains the python code used to evaluate vanilla RNN model, and this folder containes two subfolders, namely:

1) single RNN ----> evaluate single_RNN

2) Multi RNN -----> evaluate Multi_RNN

Each folder of this two subfolders contains two files, namely:


1) model_selection.py -----> The GA implementation to select model hyperparameters

2) evaluate.py ------> test RNN model

------------------------------------------------------------------------------

# Environment

OS: Ubuntu 17.10

OS type: 64-bit

USED LIBRARIES:

1- Keras (2.1.5)

2- tensorflow (1.7.0)

3- deap (1.2.2)

4- pandas (0.22.0)

5- scikit-learn (0.19.1)

6- scipy (0.18.1)

7- numpy (1.14.3)

8- matplotlib (2.0.0)









