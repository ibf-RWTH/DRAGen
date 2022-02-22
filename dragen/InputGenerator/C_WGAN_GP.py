"""
Conditional Wasserstein Generative Adversarial Network with Gradient Penalty
(C-WGAN-GP)

Author: Niklas Fehlemann, IMS IEHK
"""

# TODO: Refactor all prints to logger commands (In OutputFolder)

import sys
import torch
import torch.autograd as autograd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from tqdm import tqdm
import copy
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import seaborn as sns
import os
import time
import pickle
import datetime
from qhoptim.pyt import QHAdam

# Own Stuff
from dragen.InputGenerator import WGAN_BaseClass, gan_utils


# Inherit from the original WGAN
# Override __init__ and train
class WGANCGP(WGAN_BaseClass.WGAN):

    def __init__(self, df_list, num_features, storepath, batch_size=512, depth=2, width_d=128, width_g=128, p=0.0,
                 gen_iters=150000, learning_rate=0.00005, d_loop=5, z_dim=512,
                 embed_size=2, lambda_p=0.1, beta1=0.9, beta2=0.99,
                 n_eval=1000, optimizer='RMSProp', activationg='tanh',
                 activationd='Relu', normalize=False, centered=False, backend='tensorized'):
        if df_list.__len__() != 0:
            super(WGANCGP, self).__init__(df=df_list[0], batch_size=batch_size, num_features=num_features, depth=depth,
                                          width_d=width_d, width_g=width_g, p=p, num_epochs=None, learning_rate=learning_rate,
                                          d_loop=d_loop, z_dim=z_dim, lipschitz_const=0.01)
        else:
            super(WGANCGP, self).__init__(df=pd.DataFrame(np.random.randn(1, 4), columns=list('ABCD')), batch_size=batch_size, num_features=num_features, depth=depth,
                                          width_d=width_d, width_g=width_g, p=p, num_epochs=None,
                                          learning_rate=learning_rate,
                                          d_loop=d_loop, z_dim=z_dim, lipschitz_const=0.01)
        self.storepath = storepath
        print(self.storepath)
        self.backend = backend
        if self.backend != 'tensorized':
            print('WARNING: You`ve chosen a backend which is based on pykeops. Make sure all needed dependencies are'
                  'installed to avoid errors')
        self.loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05, reach=None, scaling=0.5, backend=self.backend)
        # Reassign some properties
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.best_states = list()
        self.normalize = normalize
        self.centered = centered
        self.n_eval = n_eval
        self.activationG = activationg
        self.activationD = activationd
        self.n_classes = df_list.__len__()
        self.embed_size = embed_size
        self.lambda_p = lambda_p
        self.beta1 = beta1
        self.beta2 = beta2
        self.gen_iters = gen_iters
        self.df_list = df_list
        self.data_list = list()  # List with single Datasets
        i = 0
        if self.df_list.__len__() != 0:
            for df in self.df_list:
                try:
                    self.data_list.append(gan_utils.GrainDataset(df=df, label=i))
                    i += 1
                except Exception as e:
                    print(e)
            self.data = torch.utils.data.ConcatDataset(self.data_list)
            self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size,
                                                          shuffle=True, num_workers=0)
        else:
            print('No initial data was provided. Please use *load_trained_states()* to add pretrained data')

        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.G = gan_utils.CGenerator(z_dim=self.z_dim, num_features=self.num_features, depth=self.depth,
                                      width=self.width_g, n_classes=self.n_classes,
                                      embed_size=self.embed_size, activation=self.activationG,
                                      normalize=self.normalize).to(self.device)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.D = gan_utils.CDiscriminator(p=self.p, num_features=self.num_features, depth=self.depth,
                                          width=self.width_d, n_classes=self.n_classes,
                                          embed_size=self.embed_size, activation=self.activationD).to(self.device)

        self.optimizer = optimizer
        if optimizer == 'Adam':
            self.optimizerG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizerD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif optimizer == 'Hyperbolic':
            self.optimizerG = QHAdam(self.G.parameters(), lr=self.lr, nus=(0.8, 1.0), betas=(self.beta1, self.beta2))
            self.optimizerD = QHAdam(self.D.parameters(), lr=self.lr, nus=(0.8, 1.0), betas=(self.beta1, self.beta2))
        elif optimizer == 'NAdam':
            self.optimizerG = QHAdam(self.G.parameters(), **QHAdam.from_nadam(lr=self.lr,
                                                                              betas=(self.beta1, self.beta2)))
            self.optimizerD = QHAdam(self.D.parameters(), **QHAdam.from_nadam(lr=self.lr,
                                                                              betas=(self.beta1, self.beta2)))
        elif optimizer == 'SGD':
            print('Beta1 is used as momentum factor. Nesterov is enabled')
            self.optimizerG = optim.SGD(self.G.parameters(), lr=self.lr, nesterov=True, momentum=self.beta1)
            self.optimizerD = optim.SGD(self.D.parameters(), lr=self.lr, nesterov=True, momentum=self.beta1)
        else:
            print('Beta1 is used as smoothing factor')
            self.optimizerG = optim.RMSprop(self.G.parameters(), lr=self.lr, alpha=self.beta1, centered=self.centered)
            self.optimizerD = optim.RMSprop(self.D.parameters(), lr=self.lr, alpha=self.beta1, centered=self.centered)
        self.SinkLosses = {}
        now = datetime.datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y_%H_%M")
        print("date and time =", self.dt_string)

    def compute_gradients(self, real_samples, fake_samples, labels):
        """
        :return: The gradient penalty for enforcing the lipschitz continuity
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), self.num_features)), device=self.device,
                             dtype=torch.float32)
        # print(alpha)
        real_samples = real_samples.to(self.device)
        labels = labels.to(self.device)
        # print(labels)
        # print(real_samples)
        # print(alpha)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).float().requires_grad_(True)
        d_interpolates = self.D(interpolates, labels)
        # print(d_interpolates.size())
        fake = torch.full(size=(real_samples.shape[0], 1), fill_value=1.0, device=self.device)
        # print(fake)
        # print(d_interpolates)
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        # print(gradients)
        gradients = gradients[0]
        # print(gradients)
        gradient_penalty = ((gradients.norm(2, dim=None) - 1) ** 2).mean()
        # print(gradient_penalty)
        return gradient_penalty

    def train(self, plot=False):
        # Write to disc
        try:
            os.mkdir(self.storepath + '/' + 'Eval_{}'.format(self.dt_string))
        except FileExistsError as e:
            print(e)
        self.write_specs()
        G_states = list()
        starttime = time.time()
        generator_iter = 0
        one = torch.tensor(1, dtype=torch.float, device=self.device)
        mone = one * -1
        while generator_iter <= self.gen_iters:
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
                    try:
                        data, labels = next(data_iter)
                    except:
                        data_iter = iter(self.dataloader)
                        data, labels = next(data_iter)

                    self.D.zero_grad()

                    # Train with real
                    D_real = self.D(data.to(self.device), labels.to(self.device))
                    D_real_loss = torch.mean(D_real)  # Work with positive values

                    D_real_loss.backward(one)
                    # Train with fakes
                    noise = torch.randn((data.__len__(), self.z_dim), device=self.device)
                    # print(noise.shape)
                    G_sample = self.G(noise, labels.to(self.device)).detach()
                    D_fake = self.D(G_sample, labels.to(self.device))
                    D_fake_loss = torch.mean(D_fake)
                    D_fake_loss.backward(mone)
                    gradient_penalty = self.compute_gradients(real_samples=data.data,
                                                              fake_samples=G_sample.data,
                                                              labels=labels.data) * self.lambda_p

                    gradient_penalty.backward()
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

                noise = torch.randn((data.__len__(), self.z_dim), device=self.device)
                G_sample = self.G(noise, labels.to(self.device))
                G_fake = self.D(G_sample, labels.to(self.device))
                G_fake_loss = torch.mean(G_fake)
                G_fake_loss.backward(one)
                self.optimizerG.step()

                generator_iter += 1
                if generator_iter % self.n_eval == 0 or generator_iter == 100:
                    # Append The states every 500 epochs
                    self.D_losses.append(D_loss.cpu().detach())
                    self.G_losses.append(G_fake_loss)
                    # Only referencing to the last g state
                    state = copy.deepcopy(self.G).cpu()
                    G_states.append(state)  # Save on CPU
                    print(
                        'Epochen {}/{}|Iterationen {}/{}| - Loss_D {}, Loss_G {}, Loss_D_real {}, '
                        'Loss_D_fake {}'
                            .format(generator_iter, self.gen_iters, i, len(self.dataloader),
                                    self.D_losses[-1],
                                    self.G_losses[-1], D_real_loss, D_fake_loss))
                    print('---')

        complete_time = time.time() - starttime
        elapsed = str(datetime.timedelta(seconds=complete_time))
        try:
            with open(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'Specs.txt', 'a') as specs:
                specs.writelines('\n')
                specs.writelines('Time {} \n'.format(elapsed))
        except:
            pass
        print('Training finished: {}'.format(elapsed))
        print('Saving Data...')
        for s in G_states:
            self.G_states.append(s.to('cpu'))
        if plot:
            if self.n_eval > 100:
                for i in range(self.n_classes):
                    for j in range(len(self.G_states)):
                        if j == 0:
                            self.plot_results(G=self.G_states[j], label=i, state=100)
                        else:
                            self.plot_results(G=self.G_states[j], label=i, state=j * self.n_eval)
            else:
                for i in range(self.n_classes):
                    for j in range(len(self.G_states)):
                        self.plot_results(G=self.G_states[j], label=i, state=j * self.n_eval)  # Plot every eval interval


        # plot Wasserstein loss
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 10))
        plt.plot(self.D_losses)
        plt.xlabel('Iterations [-]', fontsize=22)
        plt.ylabel('Discriminator loss [-]', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'Wasserstein-Loss.png')
        plt.clf()
        plt.close()
        print(os.getcwd())

    def plot_results(self, G, label, state=None):
        """
        :param state:
        :param label: Label to plot
        :return: None
        """
        noise = torch.randn((self.data_list[label][:][0].__len__(), self.z_dim), device='cpu')
        G_cpu = G.to('cpu')
        labels = torch.full((1, self.data_list[label][:][0].__len__()), fill_value=label,
                            dtype=torch.int64).view(self.data_list[label][:][0].__len__())
        res = G_cpu(noise, labels)
        df_fake_denormalized, df_real = self.normalize_data(res, label)
        df_fake_denormalized['Type'] = 'Fake'
        df_real['Type'] = 'Real'
        complete = pd.concat([df_fake_denormalized, df_real], axis=0)
        complete.reset_index(inplace=True, drop=True)
        #print(complete)

        # Plotting
        sns.set_style('darkgrid')
        sns.set(font_scale=2)
        g = sns.pairplot(complete, hue='Type', height=2.5, aspect=1, markers='o', plot_kws={"s": 50})
        for ax in g.axes[:,0]:
            ax.get_yaxis().set_label_coords(-0.45, 0.5)
        for ax in g.axes[:,1]:
            ax.get_yaxis().set_label_coords(0.5, -0.3)
        g.savefig(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'PairplotLabel_{}_{}'.format(label, state))
        plt.close()

    def evaluate(self, seeds=1):
        for label in range(self.n_classes):
            min_list = list()
            losses = list()
            for i in range(seeds):
                temp = list()
                for state in tqdm(self.G_states, desc='Iterations'):
                    # Renew RNG Seed every iter to test the states with identical
                    torch.manual_seed(i)
                    noise = torch.randn((self.data_list[label][:][0].__len__(), self.z_dim), device='cpu')
                    labels = torch.full((1, self.data_list[label][:][0].__len__()), fill_value=label,
                                        dtype=torch.int64).view(self.data_list[label][:][0].__len__())
                    fake = state(noise, labels).detach()
                    real = self.data_list[label][:][0].to('cpu')
                    fake = fake.contiguous()
                    real = real.contiguous()
                    loss = self.loss(fake, real)
                    temp.append(loss)
                # Plotting
                loss_array = np.array(temp)
                min_l = loss_array.min()
                min_i = loss_array.argmin()
                min_list.append([min_i, min_l])
                losses.append(loss_array)
                self.plot_results(self.G_states[int(min_i)], label=label,
                                  state=int(min_i) * self.n_eval if int(min_i) > 0 else 100)
            SinkLossMin = np.array(min_list)
            plt.style.use('ggplot')
            plt.figure(figsize=(10,10))
            plt.scatter(x=SinkLossMin[:, 0], y=SinkLossMin[:, 1], c='red', marker='x', s=18)
            for i in range(seeds):
                plt.annotate(text='{}'.format(round(SinkLossMin[i, 1], 7)), xy=(SinkLossMin[i, 0], SinkLossMin[i, 1]),
                             fontsize=18)
                plt.plot(losses[i])
            plt.xlabel('Iterations [-]', fontsize=22)
            plt.ylabel('Sinkhorn distance [-]', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'Sinkhorn-Loss_{}'.format(label))
            self.SinkLosses.update({label: SinkLossMin})
            plt.clf()
            plt.close()
            '''
            G_cpu = self.G_states[min_i].to('cpu')
            fake = G_cpu(noise, labels).detach()
            df_fake = pd.DataFrame(fake.data.numpy())
            df_real = self.df_list[label].dropna(axis=1)
            df_fake.columns = df_real.columns
            df_fake_denormalized = df_fake.copy()
            for feature_name in df_fake.columns:
                max_value = df_real[feature_name].max()
                min_value = df_real[feature_name].min()
                df_fake_denormalized[feature_name] = ((df_fake[feature_name] + 1) * (max_value-min_value)) / 2+min_value
            df_fake_denormalized.to_csv('Data_Label_{}.csv'.format(label), index=False)
            '''
        try:
            with open(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'Specs.txt', 'a') as specs:
                specs.writelines('\n')
                specs.writelines('Results \n')
                #print(self.SinkLosses)
                for key, value in self.SinkLosses.items():
                    specs.writelines('Label {} - Iteration {} - Result {} \n'.format(key, value[0][0], value[0][1]))
                mean = np.array(list(self.SinkLosses.values()))
                mean = np.squeeze(mean)
                mean = mean[:, 1].mean()
                specs.writelines('\n Mean Sink Error: {}'.format(mean))
        except FileNotFoundError as e:
            print(e)
        self.get_best_fit()

    def get_best_fit(self):
        # Method to find best state generator after evaluation and save it
        for key, value in self.SinkLosses.items():
            state = self.G_states[int(value[0][0])]
            self.best_states.append(state)

        if self.best_states.__len__() != self.df_list.__len__():
            print('Number of states unequal to number of dfs!')
            sys.exit(0)

        for i in tqdm(range(self.best_states.__len__())):
            with open(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'TrainedData_{}.pkl'.format(i), 'wb') as f:
                save_list = [self.best_states[i], self.df_list[i]]
                pickle.dump(save_list, f)
                f.close()

    def sample_batch(self, label, size=1000):
        # Check if there is state data present.
        if self.best_states:
            noise = torch.randn((size, self.z_dim), device='cpu')
            labels = torch.full((1, size), fill_value=label,
                                dtype=torch.int64).view(size)
            sample = self.best_states[label](noise, labels).detach()
        else:
            print('There is no state data! Aborting')
            sys.exit(0)

        # Normalize and transfer to pandas df
        sample_df, _ = self.normalize_data(sample, label)
        return sample_df

    def normalize_data(self, data, label):
        df_fake = pd.DataFrame(data.data.numpy())
        df_real = self.df_list[label].copy()
        df_fake.columns = df_real.dropna(axis=1).columns
        df_fake_denormalized = df_fake.copy()
        for feature_name in df_real.columns:
            max_value = df_real[feature_name].max()
            min_value = df_real[feature_name].min()
            df_fake_denormalized[feature_name] = ((df_fake[feature_name] + 1) * (max_value - min_value)) / 2 + min_value

        return df_fake_denormalized, df_real

    def write_specs(self):
        with open(self.storepath + '/' + 'Eval_{}/'.format(self.dt_string) + 'Specs.txt', 'w') as specs:
            # Network specs
            specs.writelines('depth: {} - width_d {} - width_g {} \n'.format(self.depth, self.width_d, self.width_g))
            specs.writelines('n_classes: {} - embed_size: {} \n'.format(self.n_classes, self.embed_size))
            specs.writelines('z_dim: {} - d_loop: {} \n'.format(self.z_dim, self.d_loop))
            specs.writelines('Dropout: {} - Features: {} \n'.format(self.p, self.num_features))
            # WGAN specs
            specs.writelines('Optimizer: {} \n'.format(self.optimizer))
            specs.writelines('learning rate: {} \n'.format(self.lr))
            specs.writelines('Batch Size: {} \n'.format(self.batch_size))
            specs.writelines('Penalty: {} \n'.format(self.lambda_p))
            specs.writelines('Iterations: {} \n'.format(self.gen_iters))
            if self.optimizer == 'Adam' or self.optimizer == 'Hyperbolic':
                specs.writelines('Betas: {},{} \n'.format(self.beta1, self.beta2))
            elif self.optimizer == 'NAdam':
                specs.writelines('Parameters {} \n'.format(QHAdam.from_nadam(lr=self.lr,
                                                                             betas=(self.beta1, self.beta2))))
            else:
                specs.writelines('Smoothing Factor: {} \n'.format(self.beta1))
            specs.writelines('Generator {} \n'.format(self.activationG))
            specs.writelines('Discriminator {} \n'.format(self.activationD))
            specs.writelines('BatchNorm? {} \n\n'.format(self.normalize))
            specs.writelines('Len of combined dataset: {}'.format(self.data.__len__()))

    def load_trained_states(self, file_list, single=False):
        # Read all the data
        if not single:
            for filepath in file_list:
                try:
                    with open(filepath, 'rb') as file:
                        data = pickle.load(file)
                        self.best_states.append(data[0])
                        self.df_list.append(data[1])
                except Exception as e:
                    print('Not all data could be loaded! \n To avoid subsequent Errors, the process is canceled')
                    print(e)
                    sys.exit(0)
        elif single:
            print('Loading deprecated BestGenerators')
            with open(file_list, 'rb') as file:
                self.best_states = pickle.load(file)


