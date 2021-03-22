import sys
import logging
import os

from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D


if __name__ == "__main__":
    last_RVE = 1  # Specify the number of iterations
    dimension = 3
    # Optional arguments with default values:
    # DataTask(box_size=50, n_pts=50, number_of_bands=0, bandwidth=3, shrink_factor=0.5, file1=None, file2=None, gui_flag=False):
    obj2D = DataTask2D()
    obj3D = DataTask3D()
    try:
        if dimension == 2:
            grains_df = obj2D.initializations(dimension)
            for i in range(last_RVE):
                obj2D.rve_generation(i, grains_df)
            sys.exit()
        elif dimension == 3:
            grains_df = obj3D.initializations(dimension)
            for i in range(last_RVE):
                obj3D.rve_generation(i, grains_df)
            sys.exit()
        else:
            LOGS_DIR = 'Logs/'
            logger = logging.getLogger("RVE-Gen")
            if not os.path.isdir(LOGS_DIR):
                os.makedirs(LOGS_DIR)
            f_handler = logging.handlers.TimedRotatingFileHandler(
                filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
            logger.setLevel(level=logging.DEBUG)
            logger.info('dimension must be 2 or 3')
            sys.exit()
    except KeyboardInterrupt:
        print('Exiting!!!')
