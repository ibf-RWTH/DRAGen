"""
File for running a sufficient CWGAN-GP Training
    INPUT: pandas DataFrames
    OUTPUTS: hdf5-files with state and input (for denormalization)
"""


import os
import datetime
import pandas as pd
from InputGenerator import C_WGAN_GP
from InputGenerator.linking import Reconstructor

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
USE THIS SCRIPT TO RUN A CWGAN-GP TRAINING:
Prerequisties:
    Needed Libraries
    Data (from EBSD for example)
    Working computer
There is no need for a gpu, although it is highly recommended.

Versions used: (as of 19.10.2023)
- Python/3.10.4
- torch==2.0.1
- geomloss==0.2.6
- matplotlib==3.4.2
- pip3 install --user geomloss'[full]'  IMPORTANT 
https://www.kernel-operations.io/geomloss/api/install.html
https://www.kernel-operations.io/keops/index.html

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++EXAMPLE++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# All files are from the dragen-Example Input
SOURCE = os.getcwd() + r'../ExampleInput/All_Inputs'

"""# Data: - all files schould have same number + sorting of columns
df1 = pd.read_csv(SOURCE + '/Input_TDxBN_AR.csv')
df2 = pd.read_csv(SOURCE + '/Input_RDxBN_AR.csv')
df3 = pd.read_csv(SOURCE + '/Input_RDxTD_AR.csv')

# Inclusions
df4 = pd.read_csv(SOURCE + '/Einschlüsse_TDxBN_AR.csv')
df5 = pd.read_csv(SOURCE + '/Einschlüsse_RDxBN_AR.csv')
df6 = pd.read_csv(SOURCE + '/Einschlüsse_RDxTD_AR.csv')

# Martensite
df7 = pd.read_csv(SOURCE + '/Input_Martensite_BNxRD_old.csv')
df8 = pd.read_csv(SOURCE + '/Input_Martensite_RDxBN_raw.csv')
df9 = pd.read_csv(SOURCE + '/Input_Martensite_RDxBN_ws.csv')"""

# Data Stainless_Steel
df1 = pd.read_csv(SOURCE + '/1_4310_TDxND_GrainData.csv')
df2 = pd.read_csv(SOURCE + '/1_4310_RDxND_GrainData.csv')
df3 = pd.read_csv(SOURCE + '/1_4310_RDxTD_GrainData.csv')

# Set up CWGAN-GP with all data
store_path = os.getcwd()
if not os.path.isdir(store_path):
    os.makedirs(store_path)

# Required parameters
df_list = [df1, df2, df3]
store_path = store_path
num_features = 6
gen_iters = 300000

# Optional parameters - use only if you know what you are doing
batch_size = 256                # Batch size
depth = 2                       # depth of both nets
width_d = 128                   # width of Discriminator
width_g = 128                   # width of Generator
p = 0.0                         # Dropout (0% seem to yield best results)
learning_rate = 0.00005         # learning rate for both (should be very slow)
d_loop = 5                      # How often D should be trained per iteration more than G (Affects train time)
z_dim = 512                     # Dimension of noise vector
embed_size = 2                  # embedding size for the different labels
lambda_p = 0.1                  # gradient penalty parameter (WARNING: Dont change, training will get unstable)
activationg = 'tanh'            # Activationfunction for G (tanh, relu, swish, lekyrelu)
activationd = 'relu'            # Activationfunction for D (tanh, relu, swish, lekyrelu)
optimizer = 'RMSProp'           # Optimizer (equal for both) (Adam, RMSProp, SDG, NAdam, Hyperbolic)
beta1 = 0.9                     # beta1 for Adam/Nadam/Hyperbolic - alpha for SGD/RMSProp
beta2 = 0.99                    # beta2 for Adam/Nadam/Hyperbolic
n_eval = 1000                   # evaluation interval (affects evaluation time)
centered = True                 # whether to compute the centered RMSProp
normalize = False               # whether to use batchNorm (false, possible bugs here)
backend = 'online'          # geomloss backend (Other than tensorized will need complete pykeops enviroment)

# Initialize the GAN-Object - Dont forget to pass the parameters :D
GAN = C_WGAN_GP.WGANCGP(df_list=df_list, storepath=store_path, num_features=num_features,
                        gen_iters=gen_iters, backend=backend)

# Train - will create a folder at "store path" were the results are stored
GAN.train(plot=False)

# evaluate using unbiased sinkhorn distance - will create "TrainedData.pkl"-files for further usage (1 file - 1 label)
GAN.evaluate()

# Sample Data from best epoch for Reconstruction-Algorithm
TDxBN = GAN.sample_batch(label=0, size=15000)
RDxBN = GAN.sample_batch(label=1, size=15000)
RDxTD = GAN.sample_batch(label=2, size=15000)

# Run the Reconstruction
Bot = Reconstructor(TDxBN, RDxBN, RDxTD, drop=True)
Bot.run(n_points=8000)  # Could take a while with 500 points...

# Plot the results as Bivariate KDE-Plots
Bot.plot_comparison(close=True)
print(Bot.result_df)
Bot.result_df.to_csv('Result.csv')

# Generate RVE-Input for given Boxsize
Bot.get_rve_input(bs=10)
print(Bot.rve_inp)   # This is the RVE-Input data
Bot.rve_inp.to_csv('RVE_inp.csv')


