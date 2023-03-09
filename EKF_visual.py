## EKF_visual ##
import torch
import numpy as np
from torch import autograd

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

class ExtendedKalmanFilter:
    def __init__(self, SystemModel,dataset_name ,prior_flag, model_encoder_trained):
        self.encoder = model_encoder_trained.double()
        self.prior_flag = prior_flag
        self.dataset_name = dataset_name

        self.f_function = SystemModel.f_function
        self.m = SystemModel.m
        self.givenQ = SystemModel.givenQ

        self.H = SystemModel.H
        self.n = SystemModel.n
        self.givenR = SystemModel.givenR

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        #self.KG_array = torch.zeros((self.T_test, self.m, self.m))

        # Full knowledge about the model or partial? (Should be made more elegant)
        # if (mode == 'full'):
        #     self.fString = 'ModAcc'
        #     self.hString = 'ObsAcc'
        # elif (mode == 'partial'):
        #     self.fString = 'ModInacc'
        #     self.hString = 'ObsInacc'

    # Predict
    def Predict(self):
        ####### Predict the 1-st moment of x [F(x)]
        if self.dataset_name == "Lorenz":
            self.F_matrix = self.f_function(self.x_t_est)
            self.x_t_given_prev = torch.matmul(self.F_matrix, self.x_t_est.float())
        else:
            self.F_matrix = autograd.functional.jacobian(self.f_function, self.x_t_est).squeeze().float()
            self.x_t_given_prev = self.f_function(self.x_t_est).transpose(0, 1)
        # Predict the 1-st moment of y  [H*F*x]
        self.y_t_given_prev = torch.matmul(self.H.double(), self.x_t_given_prev.double(), )
        # Compute the Jacobians
        self.UpdateJacobians(self.F_matrix, self.H)

        ####### Predict the 2-nd moment of x  cov(x)=[F*Cov(x)*F_T+Q]
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior.double())
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.givenQ
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.givenR

        # plt.imshow(self.y_t_given_prev)
        # plt.show()
        # plt.imshow(tmp.reshape(28,28))
        # plt.show()
        # plt.imshow(self.H.reshape(28,28))
        # plt.show()
        # Predict the 2-nd moment of y  cov(x)=[H*Cov(x)*H_T+R]

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG.double().cpu(),torch.inverse(self.m2y.cpu().detach()+ 1e-8 * np.eye(self.m)))
        # Save KalmanGain
        #self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.y_t_given_prev

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.x_t_est = self.x_t_given_prev + torch.matmul(self.KG.to(dev),self.dy.double().to(dev))
        self.x_t_est = self.x_t_est.float()
        #self.x_t_est = self.x_t_est.transpose(0, 1)
        # Compute the 2-nd posterior moment  (???)
        self.m2x_posterior = torch.matmul(self.KG.to(dev), self.m2y.double().to(dev))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.m2x_posterior.to(dev),self.KG.to(dev).transpose(0,1))

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()
        return self.x_t_est, self.m2x_posterior

    def InitSequence_EKF(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F.double()
        self.F_T = torch.transpose(F, 0, 1).double()
        self.H = H.double()
        self.H_T = torch.transpose(H, 0, 1).double()

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, video, d,target_of_video, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]) #space for estimation of state sequence
        self.encoder_output = torch.empty(size=[d, T]) #space for encoder estimation
        self.sigma = torch.empty(size=[self.m, self.m, T]) #covariance state noise
        # Pre allocate KG array
        #self.KG_array = torch.zeros((T, self.m, self.m))   # space for KG of each time step
        self.i = 0  # Index for KG_array allocation
        self.x_t_est = self.m1x_0
        #torch.squeeze(
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            obsevation_t = video[t, :, :]
            obsevation_t = torch.from_numpy(np.expand_dims(obsevation_t, axis=(0,1)))
            self.encoder.eval()
            if self.dataset_name == "Pendulum":
                if self.prior_flag:
                    encoder_output_t=torch.squeeze(self.encoder(obsevation_t.double(),self.x_t_est[0,:].unsqueeze(0).double()))
                else:
                    encoder_output_t = torch.squeeze(self.encoder(obsevation_t.double()))
                self.encoder_output[:, t] = encoder_output_t
            else:
                if self.prior_flag:
                    self.prior = torch.matmul(self.f_function(self.x_t_est), self.x_t_est)
                    encoder_output_t=self.encoder(obsevation_t.double(),self.prior.transpose(0,1).double()).transpose(0,1)
                else:
                    encoder_output_t = torch.squeeze(self.encoder(obsevation_t.double()))
                self.encoder_output[:, t] = encoder_output_t.squeeze()
            xt, sigmat = self.Update(encoder_output_t)
            self.sigma[:, :, t] = torch.squeeze(sigmat)
            self.x[:, t] = torch.squeeze(xt)
