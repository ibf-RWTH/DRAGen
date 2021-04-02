import pandas as pd
from InputGenerator.linking import Reconstructor
from InputGenerator.C_WGAN_GP import WGANCGP
import os
import datetime

SOURCE = r'../ExampleInput'
os.chdir(SOURCE)

# Data:
df1 = pd.read_csv('Input_TDxBN_AR.csv')
df2 = pd.read_csv('Input_RDxBN_AR.csv')
df3 = pd.read_csv('Input_RDxTD_AR.csv')

# Einschlüsse
df4 = pd.read_csv('Einschlüsse_TDxBN_AR.csv')
df5 = pd.read_csv('Einschlüsse_RDxBN_AR.csv')
df6 = pd.read_csv('Einschlüsse_RDxTD_AR.csv')
os.chdir('../')

# Set up CWGAN-GP with all data
store_path = 'OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(0)
if not os.path.isdir(store_path):
    os.makedirs(store_path)
GAN = WGANCGP(df_list=[df1, df2, df3, df4, df5, df6], storepath=store_path, num_features=3, gen_iters=1000)

# Run training for 5000 Its - 150.000 is default
GAN.train()

# Evaluate Afterwards
GAN.evaluate()
breakpoint()
# Sample Data from best epoch for Reconstruction-Algorithm
TDxBN = GAN.sample_batch(label=0, size=1000)
RDxBN = GAN.sample_batch(label=1, size=1000)
RDxTD = GAN.sample_batch(label=2, size=1000)

# Run the Reconstruction
Bot = Reconstructor(TDxBN, RDxBN, RDxTD, drop=True)
Bot.run(n_points=500)  # Could take a while with 500 points...

# Plot the results as Bivariate KDE-Plots
Bot.plot_comparison(close=False)

# Generate RVE-Input for given Boxsize
Bot.get_rve_input(bs=20)
print(Bot.rve_inp)   # This is the RVE-Input data
