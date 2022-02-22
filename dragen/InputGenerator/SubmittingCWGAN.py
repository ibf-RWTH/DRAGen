import pandas as pd
from dragen.InputGenerator.linking import Reconstructor
from dragen.InputGenerator.C_WGAN_GP import WGANCGP
import os
import datetime
import sys

def submit_wgan():
    SOURCE = os.getcwd() + '/GANInput'

    # Data:
    df1 = pd.read_csv(SOURCE + '/Input_TDxBN_AR.csv')
    df2 = pd.read_csv(SOURCE + '/Input_RDxBN_AR.csv')
    df3 = pd.read_csv(SOURCE + '/Input_RDxTD_AR.csv')

    # Inclusions
    df4 = pd.read_csv(SOURCE + '/Einschlüsse_TDxBN_AR.csv')
    df5 = pd.read_csv(SOURCE + '/Einschlüsse_RDxBN_AR.csv')
    df6 = pd.read_csv(SOURCE + '/Einschlüsse_RDxTD_AR.csv')

    df7 = pd.read_csv(SOURCE + '/Input_Martensit_BNxRD.csv')
    # Set up CWGAN-GP with all data
    store_path = os.getcwd() + '/Simulations/' + str(datetime.datetime.now())[:10] + \
                 '_Epoch_{}_Trial_{}_batch_{}'.format(1,
                                                      1,
                                                      1)
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
    GAN = WGANCGP(df_list=[df1, df2, df3, df4, df5, df6, df7], storepath=store_path, num_features=3, gen_iters=5000)

    # Run training for 5000 Its - 150.000 is default
    GAN.train()

    # Evaluate Afterwards
    GAN.evaluate()

    # Sample batch
    #for i in range(6):
    #    batch = GAN.sample_batch(size=5000, label=2)
    #    batch.to_csv(path_or_buf='Data.csv')

    """# Sample Data from best epoch for Reconstruction-Algorithm
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
    print(Bot.rve_inp)   # This is the RVE-Input data"""
