import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
from sklearn.neighbors import KernelDensity
from skimage.filters import threshold_minimum
import pandas as pd
import scipy.stats as st
import seaborn as sns
import numpy as np
from copy import copy
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import damask

class Texture:
    def __init__(self):
        pass



    def read_orientation(self, file: str, rve=None):

        if file[-3:] == 'odf':
            figname = 'xrd_section_plot.png'
            tex_df = pd.read_csv(file, delimiter='\s+', skiprows=5, header=None)
            tex_df.dropna(axis=1, inplace=True)
            # tex_df[2] = tex_df[2].round(0)
            """tex_df.loc[tex_df[0] > 270, 0] = 360 - tex_df.loc[tex_df[0] > 270, 0]
            tex_df.loc[tex_df[0] > 180, 0] = tex_df.loc[tex_df[0] > 180, 0] - 180
            tex_df.loc[tex_df[0] > 90, 0] = 180 - tex_df.loc[tex_df[0] > 90, 0]
            tex_df.loc[tex_df[1] > 270, 1] = 360 - tex_df.loc[tex_df[1] > 270, 1]
            tex_df.loc[tex_df[1] > 180, 1] = tex_df.loc[tex_df[1] > 180, 1] - 180
            tex_df.loc[tex_df[1] > 90, 1] = 180 - tex_df.loc[tex_df[1] > 90, 1]
            tex_df.loc[tex_df[2] > 270, 2] = 360 - tex_df.loc[tex_df[2] > 270, 2]
            tex_df.loc[tex_df[2] > 180, 2] = tex_df.loc[tex_df[2] > 180, 2] - 180
            tex_df.loc[tex_df[2] > 90, 2] = 180 - tex_df.loc[tex_df[2] > 90, 2]"""
            tex_df = tex_df.rename(columns={0: "phi1", 1: "PHI", 2: "phi2", 3: "frequency"})

        elif file[-3:] == 'txt':
            figname = 'rve_section_plot.png'
            assert (rve is not None)
            rve = np.load(rve)
            if rve.min() < 0:
                start1 = int(rve.shape[0] / 4)
                stop1 = int(rve.shape[0] / 4 + rve.shape[0] / 4 * 2)
                start2 = int(rve.shape[1] / 4)
                stop2 = int(rve.shape[1] / 4 + rve.shape[1] / 4 * 2)
                start3 = int(rve.shape[2] / 4)
                stop3 = int(rve.shape[2] / 4 + rve.shape[2] / 4 * 2)
                rve = rve[start1:stop1, start2:stop2, start3:stop3]

            grainID, counts = np.unique(rve, return_counts=True)
            tex_df = pd.read_csv(file, sep=',', header=None)
            tex_df["frequency"] = counts/(rve.shape[0]*rve.shape[1]*rve.shape[2])

            """tex_df.loc[tex_df[0] > 270, 0] = 360 - tex_df.loc[tex_df[0] > 270, 0]
            tex_df.loc[tex_df[0] > 180, 0] = tex_df.loc[tex_df[0] > 180, 0] - 180
            tex_df.loc[tex_df[0] > 90, 0] = 180 - tex_df.loc[tex_df[0] > 90, 0]
            tex_df.loc[tex_df[1] > 270, 1] = 360 - tex_df.loc[tex_df[1] > 270, 1]
            tex_df.loc[tex_df[1] > 180, 1] = tex_df.loc[tex_df[1] > 180, 1] - 180
            tex_df.loc[tex_df[1] > 90, 1] = 180 - tex_df.loc[tex_df[1] > 90, 1]
            tex_df.loc[tex_df[2] > 270, 2] = 360 - tex_df.loc[tex_df[2] > 270, 2]
            tex_df.loc[tex_df[2] > 180, 2] = tex_df.loc[tex_df[2] > 180, 2] - 180
            tex_df.loc[tex_df[2] > 90, 2] = 180 - tex_df.loc[tex_df[2] > 90, 2]"""
            tex_df = tex_df.rename(columns={0: "phi1", 1: "PHI", 2: "phi2"})

        else:
            figname = 'ebsd_section_plot.png'
            tex_df = pd.read_csv(file, dtype=float)
            tex_df.dropna(axis=1, inplace=True)
            tex_df['phi2'] = tex_df['phi2'].round(0)
            tex_df["area"] = tex_df["a"]*tex_df["b"]*np.pi
            tex_df["frequency"] = tex_df["area"]/np.sum(tex_df["area"].to_numpy())

        tex_df = tex_df[["phi1", "PHI", "phi2", "frequency"]]
        return tex_df

    def symmetry_operations(self, tex_df: pd.DataFrame, family):

        tex_df["phi1"] = tex_df["phi1"]*np.pi/180
        tex_df["PHI"] = tex_df["PHI"]*np.pi/180
        tex_df["phi2"] = tex_df["phi2"]*np.pi/180
        texture = tex_df[["phi1", "PHI", "phi2"]].to_numpy()

        symmetric_texture = damask.Orientation.from_Euler_angles(phi=texture, family=family, lattice='cI').\
            equivalent.as_Euler_angles(degrees=True)

        symmetric_texture_np = symmetric_texture.reshape(-1, 3)

        sym_tex_df = pd.DataFrame(data=symmetric_texture_np, columns=["phi1", "PHI", "phi2"])
        frequency_np = np.asarray([[tex_df.loc[i, "frequency"]/symmetric_texture.shape[0]]*symmetric_texture.shape[0]
                        for i in range(symmetric_texture.shape[1])])
        sym_tex_df["frequency"] = frequency_np.flatten()

        return sym_tex_df

    def calc_odf(self, tex_df: pd.DataFrame, phi2_list, store_path, figname):
        tex_df = tex_df[tex_df.phi1 <= 90]
        tex_df = tex_df[tex_df.PHI <= 90]
        tex_df = tex_df[tex_df.phi2 <= 90]
        phi1 = tex_df['phi1'].to_numpy()
        PHI = tex_df['PHI'].to_numpy()
        phi2 = tex_df['phi2'].to_numpy()
        weight = tex_df['frequency']
        phi1_min, PHI_min = 0, 0
        phi1_max, PHI_max = 90, 90
        # calculate KDE
        xyz = np.vstack([phi1, PHI, phi2])
        # kde = st.gaussian_kde(xyz, bw_method='scott', weights=values)
        kde = KernelDensity(kernel='gaussian', bandwidth=4).fit(xyz.transpose())

        # Evaluate kde on a grid
        n_pixel = 256
        phi1_range = np.linspace(phi1_min, phi1_max, n_pixel, endpoint=True)
        PHI_range = np.linspace(PHI_min, PHI_max, n_pixel, endpoint=True)

        fig, axs = plt.subplots(ncols=3, figsize=[16, 5.5])
        for ax, phi2 in zip(axs, phi2_list):

            xi, yi, zi = np.meshgrid(phi1_range, PHI_range, phi2, indexing='ij')
            coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
            density = np.exp(kde.score_samples(coords.transpose()))
            density = np.reshape(density, xi.shape)
            density = density[:, :, 0].reshape((n_pixel, n_pixel))

            ax.set_xlim(phi1_min, phi1_max)
            ax.set_ylim(PHI_min, PHI_max)

            cmap = plt.get_cmap("jet").copy()
            cmap.set_under('white', 0.5)

            norm = colors.LogNorm(vmin=1e-7, vmax=10e-6)
            # density plot
            im = ax.imshow(density.transpose(), cmap=cmap, norm=norm, aspect='auto',
                           extent=[phi1_min, phi1_max, PHI_min, PHI_max]
                           )

            # Contour plot
            ax.contour(xi[:, :, 0], yi[:, :, 0], np.rot90(density.transpose(), k=3), colors='white')

            title_font = {'size': 22}
            label_font = {'size': 18}

            ax.set_xlabel(r'$\varphi_1$ $(^\circ)$', fontdict=label_font)
            ax.set_ylabel(r'$\Phi$ $(^\circ)$', fontdict=label_font)
            ax.set_title(r'$\varphi_2 = {}^\circ$'.format(phi2), fontdict=title_font)

            ticks = np.arange(0, 100, 10)
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticks))

            ax.set_xticklabels(ticks, fontsize=16)
            ax.set_yticklabels(ticks, fontsize=16)
            ax.set(aspect='equal')


        cbar_ax = fig.add_axes([0.94, 0.18, 0.02, 0.61])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
        plt.subplots_adjust(left=0.05, right=0.93)

        plt.savefig(store_path+figname)
        plt.close()
        print('saved {} at {}'.format(figname, store_path))


if __name__ == '__main__':
    rve_texture = 'E:/Sciebo/IEHK/Publications/ComputationalSci/DRAGen/matdata/NO30/Texture/EulerAngles.txt'
    rve = 'E:/Sciebo/IEHK/Publications/ComputationalSci/DRAGen/matdata/NO30/Texture/RVE_Numpy.npy'
    ebsd_texture = 'E:/Sciebo/IEHK/Publications/IJPLAS/Matdata/ES_Data_processed.csv'
    xrd_texture = "E:/Sciebo/IEHK/Publications/IJPLAS/Matdata/Henrich/Henrich/MTEX.vpsc.odf"
    phi2 = [0, 45, 90]
    store_path = r"E:/Sciebo/IEHK/Publications/ComputationalSci/DRAGen/matdata/NO30/Texture/"

    files = [xrd_texture, rve_texture, ebsd_texture]
    names = {0: "xrd_section_plot.png", 1: "rve_section_plot.png", 2: "ebsd_section_plot.png"}
    for i, file in enumerate(files):
        print('starting odf of {}...'.format(names[i]))
        tex = Texture().read_orientation(file=file, rve=rve)
        sym_tex = Texture().symmetry_operations(tex_df=tex, family='cubic')
        Texture().calc_odf(sym_tex, phi2_list=phi2, store_path=store_path, figname=names[i])

