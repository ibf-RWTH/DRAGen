import sys

from dragen.main import DataTask


if __name__ == "__main__":
    last_RVE = 1  # Specify the number of iterations
    # Optional arguments with default values:
    # DataTask(box_size=22, points_on_edge=22, number_of_bands=0, bandwidth=3, speed=1, shrink_factor=0.3, file1, file2, gui_flag=False)
    obj = DataTask()
    try:
        convert_list, phase1, phase2 = obj.initializations()
        for i in range(last_RVE + 1):
            obj.rve_generation(i, convert_list, phase1, phase2)
        sys.exit()
    except KeyboardInterrupt:
        print('Exiting!!!')
