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
                obj3D = DataTask3D_GAN(box_size=20, n_pts=40, number_of_bands=n_bands, bandwidth=5, shrink_factor=0.5,
                                       band_filling=0.99, phase_ratio=0.3, inclusions_ratio=0.01,
                                       inclusions_flag=False, solver=solver, file1=None, file2=None, store_path='../',
                                       gui_flag=False, anim_flag=True, gan_flag=True, exe_flag=False)
                grains_df, store_path = obj3D.initializations(3, epoch=v)
                obj3D.rve_generation(grains_df, store_path)
                obj3D.post_processing()
                v = v + 1
                logging.shutdown()
