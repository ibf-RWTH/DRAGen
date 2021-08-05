# File for standalone running without gui
import logging
import pandas as pd
import numpy as np
from dragen.main3D_GAN import DataTask3D_GAN


if __name__ == "__main__":
    ### Instanciate some variables to vary ###

    bands = True
    solver_typ = ['Spectral']
    number_of_rve = 1

    # Read percentages:
    data_full = pd.read_csv('../ExampleInput/Full_Martensite_Percentage.csv')
    bandwidths = pd.read_csv('../ExampleInput/Bandwidths.csv')
    n_bands = pd.read_csv('../ExampleInput/NumberOfBands.csv')
    ganfile = r'../ExampleInput/BestGenerators.p'

    v = 0
    for solver in solver_typ:
        for i in range(number_of_rve):

            # Sample a martensite fraction
            idx = np.random.randint(0, data_full.__len__())
            full_percentage = data_full.iloc[idx].to_numpy()

            if bands:
                # Sample a number of bands and corresponding bandwiths
                idx = np.random.randint(0, n_bands.__len__())
                number_of_bands = 1 #int(np.round(n_bands.iloc[idx].to_numpy()*3/5))
                min_bw = 0.0
                while min_bw <= 1:
                    indices = np.random.randint(0, bandwidths.__len__(), size=number_of_bands)
                    bandwidth = bandwidths.iloc[indices].to_numpy()
                    min_bw = np.amin(bandwidth)
            else:
                number_of_bands = 0
                bandwidth = 0

            obj3D = DataTask3D_GAN(ganfile=ganfile, box_size=20, n_pts=30, number_of_bands=number_of_bands,
                                   bandwidth=bandwidth,
                                   shrink_factor=0.5,
                                   band_filling=1.3, phase_ratio=float(full_percentage), inclusions_ratio=0.01,
                                   inclusions_flag=False, solver=solver, file1=None, file2=None, store_path='../',
                                   gui_flag=False, anim_flag=True, gan_flag=True, exe_flag=False)
            grains_df, store_path = obj3D.initializations(3, epoch=v)
            obj3D.rve_generation(grains_df, store_path)
            obj3D.post_processing()
            v = v + 1
            logging.shutdown()
