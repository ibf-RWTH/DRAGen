import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity
import pandas as pd
import scipy.stats as st
import seaborn as sns

class Texture:
    def __init__(self, mat_file: str, phi2: int):
        self.file = mat_file
        self.phi2 = phi2
    def scan(self):


        if self.file[-3:] == 'odf':
            tex_df = pd.read_csv(self.file, delimiter='\s+', skiprows=5, header=None)
            tex_df.dropna(axis=1, inplace=True)
            phi1 = tex_df[0].to_numpy()
            PHI = tex_df[1].to_numpy()  # *np.pi/180
            phi2 = tex_df[2].to_numpy()  # *np.pi/180
            print(tex_df)
        else:
            tex_df = pd.read_csv(self.file, dtype=float)
            tex_df.dropna(axis=1, inplace=True)
            tex_df['phi2'] = tex_df['phi2'].round(0)
            #tex_df = tex_df.loc[tex_df['phi2'] == 45]
            phi1 = tex_df['phi1'].to_numpy()
            PHI = tex_df['PHI'].to_numpy()#*np.pi/180
            phi2 = tex_df['phi2'].to_numpy()#*np.pi/180

        phi1_min, phi1_max = 0, 90#2*np.pi
        PHI_min, PHI_max = 0, 90#np.pi/2
        phi2_min, phi2_max = 0, 90#np.pi/2

        # Peform the kernel density estimate
        xx, yy = np.mgrid[phi1_min:phi1_max:100j,
                              PHI_min:PHI_max:100j] #,
                              #phi2_min:phi2_max:50j]
        positions = np.vstack([xx.ravel(), yy.ravel(), [self.phi2]*len(xx.ravel())])
        print(positions)
        values = np.vstack([phi1, PHI, phi2])
        print(values)

        kde = KernelDensity(kernel='exponential', bandwidth=25).fit(values.transpose())
        density = np.exp(kde.score_samples(positions.transpose()))
        #print(np.exp(log_density))
        #breakpoint()
        #kernel = st.gaussian_kde(values)
        density = np.reshape(density, xx.shape)

        fig = plt.figure()
        ax = fig.gca()
        divider = make_axes_locatable(ax)
        ax.set_xlim(phi1_min, phi1_max)
        ax.set_ylim(PHI_min, PHI_max)
        # Kernel density estimate plot
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(density.transpose(), cmap='Blues', extent=[phi1_min, phi1_max, PHI_min, PHI_max])
        ax.invert_yaxis()
        fig.colorbar(im, cax=cax, orientation='vertical')
        # Contour plot
        cset = ax.contour(xx, yy, np.rot90(density.transpose(),k=3), colors='k')

        # Label plot
        # ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel(r'$\varphi_1$')
        ax.set_ylabel(r'$\Phi$')

        plt.show()


if __name__ == '__main__':
    file = '../../ExampleInput/example_block_inp.csv'
    eband = 'D:/Sciebo/IEHK/Publications/IJPLAS/Matdata/ES_Data_processed.csv'
    exdata= "D:/Sciebo/IEHK/Publications/IJPLAS/Matdata/Henrich/Henrich/MTEX.tricline.odf"
    phi2 = 45
    Texture(mat_file=exdata, phi2=phi2).scan()
