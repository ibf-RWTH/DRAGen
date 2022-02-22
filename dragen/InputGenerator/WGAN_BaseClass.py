"""
Base Class for all WGANs
Create other WGANs by inherite from the upper WGAN Class and overwrite __init__ and plot_results
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader
import torch.optim as optim
import seaborn as sns
import os
import time
import pickle
import datetime
from geomloss import SamplesLoss

from dragen.InputGenerator import gan_utils


class WGAN:

    def __init__(self, df, batch_size, num_features, depth, width_d, width_g, p, num_epochs, learning_rate, d_loop,
                 lipschitz_const, z_dim):
        torch.manual_seed(0)
        self.data = gan_utils.GrainDataset(df, label=1)
        self.batch_size = batch_size
        self.num_features = num_features
        self.depth = depth
        self.width_d = width_d
        self.width_g = width_g
        self.p = p
        self.num_epochs = num_epochs
        self.ngpu = 1
        self.lr = learning_rate
        self.d_loop = d_loop
        self.lipschitz_const = lipschitz_const
        self.z_dim = z_dim
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.G = gan_utils.Generator(self.z_dim, self.num_features, self.depth, self.width_g, self.ngpu).to(self.device)
        self.D = gan_utils.Discriminator(self.p, self.num_features, self.depth, self.width_d, self.ngpu).to(self.device)
        self.optimizerG = optim.RMSprop(self.G.parameters(), lr=self.lr)
        self.optimizerD = optim.RMSprop(self.D.parameters(), lr=self.lr)
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=0)
        # Evaluation
        # Normalized values in unit "cube"
        self.loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05, reach=None, scaling=0.5)
        self.G_losses = []
        self.D_losses = []
        self.G_states = []
        self.SinkLoss = None
        self.SinkLossMin = None

    def train(self):
        starttime = time.time()
        generator_iter = 0
        one = torch.tensor(1, dtype=torch.float)
        mone = -1 * one
        D_loss = 0.001

        for epoch in range(self.num_epochs):
            data_iter = iter(self.dataloader)
            i = 0

            while i < len(self.dataloader):
                #####################################
                # Update D first, train D more the G
                #####################################

                # Reactivate the gradients
                for p in self.D.parameters():
                    p.requires_grad = True

                # train the discriminator more if the generator has less training
                if generator_iter < 25:
                    loop = 100
                else:
                    loop = self.d_loop
                j = 0

                while j < loop and i < len(self.dataloader):

                    data, _ = data_iter.next()
                    # clamp the parameters
                    for p in self.D.parameters():
                        p.data.clamp_(-self.lipschitz_const, self.lipschitz_const)
                    self.D.zero_grad()

                    # Train with real
                    D_real = self.D(data.to(self.device))
                    D_real_loss = torch.mean(D_real)  # Work with positive values
                    D_real_loss.backward(one)

                    # Train with fakes
                    noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
                    G_sample = self.G(noise).detach()
                    D_fake = self.D(G_sample)
                    D_fake_loss = torch.mean(D_fake)
                    D_fake_loss.backward(mone)

                    D_loss = D_fake_loss - D_real_loss  # The real Wasserstein-loss
                    self.optimizerD.step()
                    j += 1
                    i += 1

                ############################
                # Train the generator
                ############################
                for p in self.D.parameters():
                    p.requires_grad = False
                self.G.zero_grad()

                noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
                G_sample = self.G(noise)
                G_fake = self.D(G_sample)
                G_fake_loss = torch.mean(G_fake)
                G_fake_loss.backward(one)
                self.optimizerG.step()

                generator_iter += 1
                if generator_iter % 500 == 0:
                    # Append The states every 500 epochs
                    self.D_losses.append(D_loss)
                    self.G_losses.append(G_fake_loss)
                    # Only referencing to the last g state
                    state = copy.deepcopy(self.G.to('cpu'))
                    self.G_states.append(state)  # Save on CPU
                    # Calc sinkhorn
                    print(
                        'Epochen {}/{}|Iterationen {}/{}|Gen_it {} - Loss_D {}, Loss_G {}, Loss_D_real {}, '
                        'Loss_D_fake {}'
                        .format(epoch, self.num_epochs, i, len(self.dataloader), generator_iter,
                                self.D_losses[-1],
                                self.G_losses[-1], D_real_loss, D_fake_loss))
                    print('---')

        complete_time = time.time() - starttime
        elapsed = str(datetime.timedelta(seconds=complete_time))
        print('Training finished: {}'.format(elapsed))
        print('Saving Data...')
        try:
            os.mkdir('States')
        except FileExistsError as fee:
            print(fee)
        with open('States/States_{}_{}.p'.format(self.width, self.depth), 'wb') as f:
            pickle.dump(self.G_states, f)

    def evaluate(self, kind='seed'):
        # Evalutes the GAN with FIXED noises!
        if kind == 'seed':
            min_list = list()
            losses = list()
            for i in range(5):
                temp = list()
                for state in self.G_states:
                    # Renew RNG Seed every iter to test the states with identical
                    torch.manual_seed(i)
                    noise = torch.randn((self.data.__len__(), self.z_dim), device=self.device)
                    fake = state(noise).detach()
                    real = self.data[:][0].to('cpu')
                    loss = self.loss(fake, real)
                    temp.append(loss)
                # Plotting
                loss_array = np.array(temp)
                min_l = loss_array.min()
                min_i = loss_array.argmin()
                min_list.append([min_i, min_l])
                losses.append(loss_array)
            self.SinkLossMin = np.array(min_list)
            self.SinkLoss = np.array(losses)
            plt.title(label='Sinkhorn-Losses_{}'.format(kind))
            plt.scatter(x=self.SinkLossMin[:, 0], y=self.SinkLossMin[:, 1], c='red', marker='x', s=15)
            plt.plot(losses[0])
            plt.plot(losses[1])
            plt.plot(losses[2])
            plt.savefig('Sinkhorn-Loss')
            plt.show()

    def plot_results(self):
        noise = torch.randn((self.data.__len__(), 256), device='cpu')
        G_cpu = self.G.to('cpu')
        res = G_cpu(noise)
        df_fake = pd.DataFrame(res.data.numpy())
        df_real = pd.DataFrame(self.data[:][0].data.numpy())
        df_fake['Type'] = 'Fake'
        df_real['Type'] = 'Real'
        complete = pd.concat([df_fake, df_real], axis=0)
        complete.reset_index()
        sns.pairplot(complete, hue='Type')
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('Input_RDxTD.csv')
    # print(df)
    Wgan = WGAN(df, 32, 4, 1, 256, 0.5, 500, 0.00005, 5, 0.01, 512)
    Wgan.train()
    Wgan.evaluate()
    print(Wgan.SinkLossMin)

