import sys

from dragen.main2D import DataTask2D


if __name__ == "__main__":
    last_RVE = 1  # Specify the number of iterations
    dimension = 2
    # Optional arguments with default values:
    # DataTask(box_size=22, points_on_edge=22, number_of_bands=0, bandwidth=3, speed=1, shrink_factor=0.3, file1, file2, gui_flag=False)
    obj2D = DataTask2D()
    obj3D = DataTask2D()
    try:
        if dimension == 2:
            grains_df = obj2D.initializations()
            for i in range(last_RVE):
                obj2D.rve_generation(i, grains_df)
            sys.exit()
        else:
            grains_df = obj2D.initializations()
            for i in range(last_RVE):
                obj3D.rve_generation(i, grains_df)
            sys.exit()
    except KeyboardInterrupt:
        print('Exiting!!!')
