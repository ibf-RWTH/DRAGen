import pandas as pd
import matplotlib.pyplot as plt
from dragen.generation.mesh_subs import SubMesher
from dragen.utilities.Helpers import RveInfo

if __name__ == "__main__":

    subs_rve = pd.read_csv(r"F:\codes\DRAGen\OutputData\2022-11-16_000\substruct_data.csv")
    num_blocks = subs_rve['block_id'].max()
    print("num of blocks is, ", num_blocks)
    ax = plt.figure().add_subplot(111, projection='3d')
    for i in range(1, num_blocks + 1):
        block = subs_rve.loc[subs_rve['block_id'] == i]
        ax.scatter(block['x'], block['y'], block['z'])
    plt.show()
