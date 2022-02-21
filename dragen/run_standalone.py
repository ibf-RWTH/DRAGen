# File for standalone running without gui
import logging
from dragen.main3D_GAN import DataTask3D_GAN


if __name__ == "__main__":
    ### Instanciate some variables to vary ###

    number_of_bands = [0, 1]
    solver_typ = ['FEM']
    number_of_rve = 2

    v = 0
    for n_bands in number_of_bands:
        for solver in solver_typ:
            for i in range(number_of_rve):
                obj3D = DataTask3D_GAN(band_filling=0.99, inclusions_ratio=0.01, solver=solver, inclusions_flag=False)
                grains_df, store_path = obj3D.initializations(3, epoch=v)
                obj3D.rve_generation()
                obj3D.post_processing()
                v = v + 1
                logging.shutdown()
