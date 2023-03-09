## KalmanNet_nn_visual_new_architecture ##
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
#from config import F, m
import time

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

in_mult = 1
out_mult = 1

class KalmanNetLatentNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self,dataset_name):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name
    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, SysModel, model_encoder, fix_encoder_flag,d):
        self.fix_encoder_flag = fix_encoder_flag
        self.InitSystemDynamics(SysModel.f_function,SysModel.n, SysModel.H,SysModel.m, model_encoder)
        self.InitSequence(SysModel.m1x_0, SysModel.T)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)
        self.f_function = SysModel.f_function
        self.H = SysModel.H
        self.d = d
    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S):
        self.seq_len_input = 1
        self.batch_size = 1
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).double()
        self.h_Q = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q).to(dev, non_blocking=True)
        #self.batch_size
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).double()
        self.h_Sigma = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).to(dev, non_blocking=True)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n ** 2

        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).double()
        self.h_S = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S).to(dev, non_blocking=True)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        if self.dataset_name=="pendulum":
            self.d_output_FC2 = self.n
        else:
            self.d_output_FC2 = self.n*self.n
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()).double()

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU())

        # Fully connected 7
        if self.dataset_name == "pendulum":
            self.d_input_FC7 = self.n
        else:
            self.d_input_FC7 = self.n*2
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU())

        """
        # Fully connected 8
        self.d_input_FC8 = self.d_hidden_Q
        self.d_output_FC8 = self.d_hidden_Q
        self.d_hidden_FC8 = self.d_hidden_Q * Q_Sigma_mult
        self.FC8 = nn.Sequential(
                nn.Linear(self.d_input_FC8, self.d_hidden_FC8),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC8, self.d_output_FC8))
        """

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f_function,n, H ,m ,model_encoder):

        # if (infoString == 'partialInfo'):
        #     self.fString = 'ModInacc'
        #     self.hString = 'ObsInacc'
        # else:
        #     self.fString = 'ModAcc'
        #     self.hString = 'ObsAcc'

        ####### Set State Evolution Function
        self.f_function = f_function
        self.m = m

        ####### Set Observation Function
        self.H = H
        self.n = m #int(np.sqrt(n))  too small to learn?

        ###### set encoder
        self.model_encoder = model_encoder

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        self.T = T
        self.x_out = torch.empty(self.m, T).to(dev, non_blocking=True)
        self.m1x_posterior = M1_0.to(dev, non_blocking=True)
        self.m1x_posterior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.m1x_prior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.y_previous = torch.matmul(self.H,self.m1x_posterior.float())
            #.reshape(self.m, 1)
        self.i = 0

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        if self.dataset_name == "Lorenz":
            self.m1x_prior = torch.matmul(self.f_function(self.m1x_posterior.float()),self.m1x_posterior.float())
        else:
            self.m1x_prior = self.f_function(self.m1x_posterior.float()).transpose(0,1)
        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H,self.m1x_prior.float())


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_posterior_previous)
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.d)).double()

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()
        # Compute Kalman Gain
        self.step_KGain_est(y)
        # Innovation
        dy = y - self.m1y.squeeze(1)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy.double()).unsqueeze(1)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        # self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior
        # update y_prev
        self.y_previous = y

        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        if self.dataset_name == "Pendulum":
            obs_diff = obs_diff.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            obs_innov_diff = obs_innov_diff.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            obs_diff = expand_dim(obs_diff)
            obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = fw_evol_diff.double()
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5.double()
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q.double())

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff.double()
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2).double()
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma.double())

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2).double()
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S.double())

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(dev, non_blocking=True)
        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        #self.N_samples=N_samples
        self.x_out = self.KNet_step(y)

        return self.x_out

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()