import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from sklearn.neighbors import KernelDensity
import copy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import damask

class Texture:
    def __init__(self):
        pass

    def read_orientation(self,  experimentalData=None, rve_np=None, rve_df=None):

        tex_dict = dict()
        ex_tex_df = experimentalData
        ex_tex_df.dropna(axis=1, inplace=True)
        ex_tex_df['phi2'] = ex_tex_df['phi2'].round(0)
        ex_tex_df["area"] = ex_tex_df["a"]*ex_tex_df["b"]*np.pi
        ex_tex_df["frequency"] = ex_tex_df["area"]/np.sum(ex_tex_df["area"].to_numpy())
        ex_tex_df = ex_tex_df[["phi1", "PHI", "phi2", "frequency"]]
        tex_dict['experimental'] = ex_tex_df

        grainID, counts = np.unique(rve_np, return_counts=True)
        rve_tex_df = rve_df

        rve_tex_df["frequency"] = counts[rve_tex_df['GrainID'].values]/(rve_np.shape[0]*rve_np.shape[1]*rve_np.shape[2])
        rve_tex_df.dropna(axis=1, inplace=True)
        rve_tex_df['phi2'] = rve_tex_df['phi2'].round(0)
        rve_tex_df = rve_tex_df[["phi1", "PHI", "phi2", "frequency"]]
        tex_dict['rve'] = rve_tex_df
        return tex_dict

    def symmetry_operations(self, tex_df: pd.DataFrame, family):

        tex_df["phi1"] = tex_df["phi1"]*np.pi/180
        tex_df["PHI"] = tex_df["PHI"]*np.pi/180
        tex_df["phi2"] = tex_df["phi2"]*np.pi/180
        texture = tex_df[["phi1", "PHI", "phi2"]].to_numpy()

        symmetric_texture = damask.Orientation.from_Euler_angles(phi=texture, family=family, lattice='cI').\
            equivalent.as_Euler_angles(degrees=True)

        symmetric_texture_np = symmetric_texture.reshape(-1, 3)
        tex_df = tex_df.reset_index(drop=True)
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
        kde = KernelDensity(kernel='gaussian', bandwidth=4).fit(xyz.transpose(), sample_weight=weight)

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
            
            local_cmap = copy.copy(plt.get_cmap("jet"))
            local_cmap.set_under('white', 0.5)

            norm = colors.LogNorm(vmin=1e-7, vmax=10e-6)
            # density plot
            im = ax.imshow(density.transpose(), cmap=local_cmap, norm=norm, aspect='auto',
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

        plt.savefig(store_path+'/'+figname)
        plt.close()
        print(f'saved {figname} at {store_path}')

