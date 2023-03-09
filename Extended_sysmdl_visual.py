## Extended_sysmdl_visual ##
import torch
from config_script import sinerio
from model_Lorenz import y_size, m
import numpy as np

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    cuda0 = torch.device("cpu")
    #print("Running on the CPU")

class SystemModel:
    def __init__(self, f_function, given_q, real_q2, m, h_function, H, given_r, real_r, n, T, T_test, dataset_name, prior_Q=None, prior_Sigma=None, prior_S=None):
        ####################
        ### Motion Model ###
        ####################
        self.dataset_name = dataset_name
        self.f_function = f_function
        self.real_q2 = real_q2
        self.m = m
        #self.realQ = real_q * real_q * torch.eye(self.m)
        #self.givenQ = given_q * given_q * torch.eye(self.m)
        # if self.modelname == 'pendulum':
        #     self.Q = q * q * torch.tensor([[(delta_t ** 3) / 3, (delta_t ** 2) / 2],
        #                                    [(delta_t ** 2) / 2, delta_t]])
        # elif self.modelname == 'pendulum_gen':
        #     self.Q = q * q * torch.tensor([[(delta_t_gen ** 3) / 3, (delta_t_gen ** 2) / 2],
        #                                    [(delta_t_gen ** 2) / 2, delta_t_gen]])
        # else:
        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.h_function = h_function
        self.n = n
        #self.realR = real_r * real_r * torch.eye(self.n)
        #self.givenR = given_r * given_r * torch.eye(m)

        # Assign T and T_test
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.m)
                #self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        #torch.squeeze(m1x_0).to(cuda0)
        self.m2x_0 = torch.squeeze(m2x_0).to(cuda0)

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def Generate_states(self, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0.double()
        #print("*************")
        # Generate Sequence Iteratively
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            # Process Noise
            xt = torch.matmul(self.f_function(self.x_prev).double(), self.x_prev)
            mean = torch.zeros([self.m])
            # if self.modelname == "pendulum":
            #     distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
            #     eq = distrib.rsample()
            # else:
            eq = torch.normal(mean, np.sqrt(self.real_q2))
            #print(eq)

            # Additive Process Noise
            xt = torch.add(xt, eq.unsqueeze(1))

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    def generate_observations(self, T):
        ################
        ### Emission ###
        ################
        # Pre allocate an array for current observation
        self.y = np.empty(shape=[T,y_size,y_size])
        for t in range(T):
            xt=self.x[:,t]
            yt = self.h_function(xt)

            #if k%202==0:
            #   imgplot = plt.imshow(yt.cpu().detach().numpy())
            #   plt.title(xt)
            #   plt.colorbar()
            #   plt.show()

            #yt = yt.reshape(y_size * y_size)

            # Observation Noise
            #mean = torch.zeros([self.n])
            #er = torch.normal(mean, self.real_r)

            # er = np.random.multivariate_normal(mean, R_gen, 1)
            # er = torch.transpose(torch.tensor(er), 0, 1)

            # Additive Observation Noise
            #yt = torch.add(yt, er)

            #if k%202==0:
            #   imgplot = plt.imshow(yt.reshape((28,28)).cpu().detach().numpy())
            #   plt.title(xt)
            #   plt.colorbar()
            #   plt.show()

            ########################
            ### Squeeze to Array ###
            ########################
            # Save Current Observation to Trajectory Array
            self.y[t,:] = yt
            itay=29

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, T, y_size,y_size)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        initConditions = self.m1x_0

        ### Generate Examples
        for i in range(0, size):
            # Generate Sequence
            # Randomize initial conditions to get a rich dataset
            if (randomInit):
                initConditions = torch.rand_like(self.m1x_0) * variance
            self.InitSequence(initConditions, self.m2x_0)
            if sinerio == "Decimation":
                self.Generate_states(T*20)
                decimated_states = torch.empty(size=[self.m, T])
                for k in range (T*20):
                    if k % 20 == 0:
                        decimated_states[:, int(k / 20)] = self.x[:, k]
                self.x = decimated_states
            else:
                self.Generate_states(T)

            #self.x=self.transform_to_range(self.x)

             # import matplotlib.pyplot as plt
             # fig = plt.figure()
             # ax = fig.gca(projection="3d")
             # #plt.axis('off')
             # plt.grid(True)
             # ax.plot(self.x[0,:], self.x[1,:], self.x[2,:])
             # ax.set_yticklabels([])
             # ax.set_xticklabels([])
             # ax.set_zticklabels([])
             # plt.draw()
             # plt.show()

            self.generate_observations(T)

            # Training sequence input
            if np.isnan(self.y).any():
                itay = 29
            self.Input[i, :, :,:] = torch.from_numpy(self.y)
            # Training sequence output
            self.Target[i, :, :] = self.x
            print(i)

    def transform_to_range(self,x):
        x_transformed = torch.empty(self.m, x.shape[1])
        x1=x[0,:]
        x1=(x1-x1.min()+1)/(x1.max()-x1.min())*25
        x2=x[1,:]
        x2=(x2-x2.min()+1)/(x2.max()-x2.min())*25
        x_transformed[0,:] = x1
        x_transformed[1,:] = x2
        x_transformed[2,:] = x[2,:]
        return x_transformed


