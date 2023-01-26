from cProfile import label
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input data
    sub_data = pd.read_csv(
        r"/home/doelz-admin/projects/SubstructJulia/DRAGen/ExampleInput/example_block_inp.csv")
    print(sub_data)
    kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                        param_grid={"bandwidth": bandwidths},
                        cv=kFold)

    # fit input distribution
    bts = sub_data['block_thickness'].to_numpy().reshape(-1, 1)
    grid.fit(bts)
    estimator = grid.best_estimator_
    xs = np.linspace(sub_data['block_thickness'].min(), sub_data['block_thickness'].max(), 100)
    xs = np.sort(xs)
    xs = xs.reshape(-1, 1)
    ys = np.exp(estimator.score_samples(xs))
    print("finish fitting input distribution")
    #fit generated distribution
    data = pd.read_csv(r"dragen/test/substruct_data.csv")
    data = data.groupby("block_id").head()
   
    bts2 = data['block_thickness'].to_numpy().reshape(-1, 1)
    grid.fit(bts2)
    print("finish fitting generated distribution")
    estimator2 = grid.best_estimator_
    xs2 = np.linspace(data['block_thickness'].min(), data['block_thickness'].max(), 100)
    xs2 = np.sort(xs2)
    xs2 = xs2.reshape(-1, 1)
    ys2 = np.exp(estimator.score_samples(xs2))
    plt.plot(xs2, ys2,label="generated block thickness distribution")
    plt.plot(xs, ys,label="input block thickness distribution")
    plt.xlabel("Block Thickness")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
