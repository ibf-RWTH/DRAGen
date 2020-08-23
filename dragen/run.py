import sys

from dragen.main import DataTask

if __name__ == "__main__":
    last_RVE = 10
    obj = DataTask()
    try:
        convert_list, phase1, phase2 = obj.initializations()
        for i in range(last_RVE + 1):
            obj.rve_generation(i, convert_list, phase1, phase2)
        sys.exit()
    except KeyboardInterrupt:
        print('Exiting!!!')
