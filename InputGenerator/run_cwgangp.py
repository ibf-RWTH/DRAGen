"""
File for running a sufficient CWGAN-GP Training
    INPUT: pandas DataFrames
    OUTPUTS: hdf5-files with state and input (for denormalization)
"""


import os
import datetime
import pandas as pd
from InputGenerator import C_WGAN_GP

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
USE THIS SCRIPT TO RUN A CWGAN-GP TRAINING:
Prerequisties:
    Needed Libraries
    Data (from EBSD for example)
    Working computer
There is no need for a gpu, although it is highly recommended.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++EXAMPLE++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# All files are from the dragen-Example Input
SOURCE = r'C:\Venvs\dragen\ExampleInput'

# Data: - all files schould have same number + sorting of columns
df1 = pd.read_csv(SOURCE + '/Input_TDxBN_AR.csv')
df2 = pd.read_csv(SOURCE + '/Input_RDxBN_AR.csv')
df3 = pd.read_csv(SOURCE + '/Input_RDxTD_AR.csv')

# Inclusions
df4 = pd.read_csv(SOURCE + '/Einschlüsse_TDxBN_AR.csv')
df5 = pd.read_csv(SOURCE + '/Einschlüsse_RDxBN_AR.csv')
df6 = pd.read_csv(SOURCE + '/Einschlüsse_RDxTD_AR.csv')

# Martensite
df7 = pd.read_csv(SOURCE + '/Input_Martensit_BNxRD.csv')

# Set up CWGAN-GP with all data
store_path = 'C:/Venvs/dragen/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(0)
if not os.path.isdir(store_path):
    os.makedirs(store_path)

# Required parameters
df_list = [df1, df2, df3, df4, df5, df6, df7]
store_path = store_path
num_features = 3
gen_iters = 100

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
backend = 'tensorized'          # geomloss backend (Other than tensorized will need complete pykeops enviroment)

# Initialize the GAN-Object
GAN = C_WGAN_GP.WGANCGP(df_list=df_list, storepath=store_path, num_features=num_features,
                        gen_iters=gen_iters)

# Train - will create a folder at "store path" were the results are stored
GAN.train(plot=False)

# evaluate using unbiased sinkhorn distance - will create "TrainedData.pkl"-files for further usage (1 file - 1 label)
GAN.evaluate()





