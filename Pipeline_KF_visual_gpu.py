## Pipeline_KF_visual ##
import torch
import torch.nn as nn
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

class Pipeline_KF:
    def __init__(self, Time, path_KNetLatent, dataset_name, sinerio, fix_encoder_flag, ssModel, d, warm_start_flag):
        super().__init__()
        self.Time = Time # Time date and clock
        self.sinerio = sinerio
        self.path_KNetLatent = path_KNetLatent
        self.dataset_name = dataset_name # is it pendulum data?
        self.fix_encoder_flag = fix_encoder_flag     # is the encoder fix?
        self.ssModel = ssModel
        self.warm_start_flag = warm_start_flag
        self.d =d

    def setLearnedModel(self, KNet_model):
        self.model = KNet_model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def check_tracking(self, ti, knet, target,  EKF, encoder_output, folder_learning_path):
        ## Check Changes
        if self.dataset_name == "Lorenz":
            fig, (ax1, ax2, ax3) = plt.subplots(3)
        else:
            fig, (ax1, ax2) = plt.subplots(2)

        fig.suptitle("components Estimation in epoch {}".format(ti))

        KNet_est_0 = knet[0, :].cpu().detach().numpy()
        GT_0 = target[0, :]
        Enc_0 = encoder_output[0, :].cpu().detach().numpy()
        EKF_0 = EKF[0, :].cpu().detach().numpy()
        axis = list(range(GT_0.shape[0]))
        ax1.plot(axis, GT_0)
        ax1.plot(axis, KNet_est_0)
        ax1.plot(axis, Enc_0)
        ax1.plot(axis, EKF_0)
        ax1.legend(["GT", "KNet", "Encoder", "EKF"])

        KNet_est_1 = knet[1, :].cpu().detach().numpy()
        GT_1 = target[1, :]
        Enc_1 = encoder_output[1, :].cpu().detach().numpy()
        EKF_1 = EKF[1, :].cpu().detach().numpy()
        axis = list(range(GT_1.shape[0]))
        ax2.plot(axis, GT_1)
        ax2.plot(axis, KNet_est_1)
        ax2.plot(axis, Enc_1)
        ax2.plot(axis, EKF_1)
        ax2.legend(["GT", "KNet", "Encoder", "EKF"])

        if self.dataset_name == "Lorenz":
            KNet_est_2 = knet[2, :].cpu().detach().numpy()
            GT_2 = target[2, :]
            Enc_2 = encoder_output[2, :].cpu().detach().numpy()
            EKF_2 = EKF[2, :].cpu().detach().numpy()
            axis = list(range(GT_2.shape[0]))
            ax3.plot(axis, GT_2)
            ax3.plot(axis, KNet_est_2)
            ax3.plot(axis, Enc_2)
            ax3.plot(axis, EKF_2)
            ax3.legend(["GT", "KNet", "Encoder", "EKF"])

        fig.show()
        fig.savefig(folder_learning_path + "/1D epoch {}.png".format(ti))

        if self.dataset_name == "Lorenz":
            fig, ax = plt.subplots(2,2,figsize=(10,10),subplot_kw=dict(projection='3d'))
            fig.suptitle("components 3D comparison in epoch {}".format(ti))
            KNet_3D = ax[0,0].plot(KNet_est_0, KNet_est_1, KNet_est_2)
            ax[0,0].set_title("KNet")
            Enc_3D = ax[0,1].plot(Enc_0, Enc_1, Enc_2)
            ax[0,1].set_title("Encoder")
            GT_3D = ax[1,0].plot(GT_0, GT_1, GT_2)
            ax[1,0].set_title("GT")
            EKF_3D = ax[1,1].plot(EKF_0, EKF_1, EKF_2)
            ax[1,1].set_title("EKF with Encoder")
            fig.show()
            fig.savefig(folder_learning_path + "/3D epoch {}.png".format(ti))

    def get_seq_knet_output(self,observation_seq , target, prior_flag):
        if self.warm_start_flag:
            length_seq = observation_seq.shape[1]
        else:
            length_seq = observation_seq.shape[0]
        x_out = torch.empty(self.ssModel.m, length_seq)
        z_encoder_output = torch.empty(self.d, length_seq)
        if self.dataset_name == "Pendulum":
            state_prev = torch.from_numpy(np.array([90* torch.pi / 180,0])).unsqueeze(1)
        else:
            state_prev = torch.ones(self.ssModel.m,1)
        for t in range(0, length_seq):
            if self.warm_start_flag:
                y_decoaded = torch.from_numpy(observation_seq[:,t])[0]
                z_encoder_output[:, t] = y_decoaded.clone()
            else:
                AE_input = torch.from_numpy(np.expand_dims(observation_seq[t,:,:], axis=(0,1)))
                self.model.model_encoder = self.model.model_encoder.float()
                if prior_flag:
                    if self.dataset_name == "Pendulum":
                        prior = self.model.f_function(state_prev.clone()).clone()
                        y_decoaded = self.model.model_encoder(AE_input.float(),prior[:, 0].unsqueeze(0).float()).squeeze().clone()
                        state_prev = prior
                        state_prev[:, 0] = y_decoaded
                    else:
                        prior = torch.matmul(self.model.f_function(state_prev).float(), state_prev).transpose(0,1)
                        y_decoaded = self.model.model_encoder(AE_input.float(),prior.float()).squeeze().clone()
                        state_prev = y_decoaded.unsqueeze(0)
                else:
                    y_decoaded = self.model.model_encoder(AE_input.float()).squeeze()
                z_encoder_output[:, t] = y_decoaded.clone()
            x_out[:,t]= self.model(y_decoaded.clone()).clone()
            state_prev = state_prev.transpose(0, 1)
        return x_out, z_encoder_output

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target, title, prior_flag):
        ####################### freezing weights ############################
        if self.fix_encoder_flag:                  ## encoder weights are freezed - only KGain trained
            for param in self.model.parameters():
                param.requires_grad = True          ## all weights are trainable
            #if self.fix_encoder_flag:               ## encoder weights are freezed
            #    for param in self.model.model_encoder.parameters():
            #        param.requires_grad = False
        else:                                       ## encoder weights should be trained - KGain freezed
            for param in self.model.parameters():
                param.requires_grad = False         ## all weights are freezed
            if not self.fix_encoder_flag:
                for param in self.model.model_encoder.parameters():
                    param.requires_grad = True      ## encoder weights are trainable

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Knet Pipeline include {} trainable parameters".format(pytorch_total_params))
        pytorch_encoder_params = sum(p.numel() for p in self.model.model_encoder.parameters() if p.requires_grad)
        print("Knet Pipeline include {} trainable parameters".format(pytorch_encoder_params))

        ################## CV space declarations ##########################
        self.N_CV = n_CV
        ## tracking optimal validation value ##
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        ## linear MSE for each batch (samples in batch) in CV ##
        MSE_cv_linear_batch = torch.empty([self.N_CV])
        ## linear MSE for each epoch (all samples in CV) ##
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        ## log MSE for each epoch (all samples in CV) ##
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        ## linear MSE for each batch (samples in batch) in CV for encoder##
        MSE_cv_linear_batch_encoder = torch.empty([self.N_CV])
        ## linear MSE for each epoch (all samples in CV) for encoder##
        self.MSE_cv_linear_epoch_encoder = torch.empty([self.N_Epochs])
        ## log MSE for each epoch (all samples in CV) for encoder##
        self.MSE_cv_dB_epoch_encoder = torch.empty([self.N_Epochs])
        ####################################################

        ##############
        ### Epochs ###
        ##############

        Train_loss_list=[]
        Val_loss_list = []
        for ti in range(0, self.N_Epochs):
            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.eval()
            ################ running on each sample ########
            for j in range(0, self.N_CV):
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)
                y_cv = cv_input[j, :, :]
                x_out_cv, z_encoder_output_cv = self.get_seq_knet_output(y_cv, cv_target[j, :, :],prior_flag)
                # Compute Training Loss
                if self.dataset_name =="Pendulum":
                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv[0,:], cv_target[j, 0, :]).detach()
                    MSE_cv_linear_batch_encoder[j] = self.loss_fn(z_encoder_output_cv.float(),cv_target[j, 0, :].float())
                else:
                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).detach()
                    MSE_cv_linear_batch_encoder[j] = self.loss_fn(z_encoder_output_cv.float(), cv_target[j, :, :].float())
                if MSE_cv_linear_batch[j].isnan() == True:
                    Itay = 29
                    MSE_cv_linear_batch[j] = 1
                    print("**** we have nan value ****")
                    #break
                if j == 4:
                    print("encoder output {} x state {} kalman output {}".format(z_encoder_output_cv[:, 10], cv_target[j, :, 10],x_out_cv[:, 10]))
            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            self.MSE_cv_linear_epoch_encoder[ti] = torch.mean(MSE_cv_linear_batch_encoder)
            self.MSE_cv_dB_epoch_encoder[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch_encoder[ti])

            # saving if better than optimal weights
            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.path_KNetLatent)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            ################## train space declarations ##########################
            self.N_E = n_Examples
            ## linear MSE for each batch (samples in batch) in train ##
            MSE_train_linear_batch = torch.empty([self.N_B])
            ## linear MSE for each epoch (all samples in train) ##
            self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
            ## log MSE for each epoch (all samples in train) ##
            self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

            ## linear MSE for each batch (samples in batch) in train for encoder##
            MSE_train_linear_batch_encoder = torch.empty([self.N_B])
            ## linear MSE for each epoch (all samples in train) encoder##
            self.MSE_train_linear_epoch_encoder = torch.empty([self.N_Epochs])
            ## log MSE for each epoch (all samples in train) encoder##
            self.MSE_train_dB_epoch_encoder = torch.empty([self.N_Epochs])
            ########################################################################

            # Training Mode
            #self.model.train()
            # Init Hidden State
            self.model.init_hidden()
            Batch_Optimizing_LOSS_sum = 0
            Batch_Optimizing_LOSS_sum_Encoder = 0

            for j in range(0, self.N_B):
                print(j)
                n_e = random.randint(0, self.N_E - 1)
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)
                y_training = train_input[n_e, :, :] # chosen trajectory to learn from
                x_out_training, z_encoder_output_train = self.get_seq_knet_output(y_training, train_target[n_e, :, :], prior_flag)
                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training.float(), train_target[n_e, :, :].float())
                LOSS_Encoder = self.loss_fn(z_encoder_output_train.float(), train_target[n_e, :, :].float())
                MSE_train_linear_batch[j] = LOSS.detach()
                MSE_train_linear_batch_encoder[j] = LOSS_Encoder.detach()
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS
                Batch_Optimizing_LOSS_sum_Encoder = Batch_Optimizing_LOSS_sum_Encoder + LOSS_Encoder
            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti]).cpu().detach().numpy()

            self.MSE_train_linear_epoch_encoder[ti] = torch.mean(MSE_train_linear_batch_encoder)
            self.MSE_train_dB_epoch_encoder[ti] = 10 * torch.log10(self.MSE_train_linear_epoch_encoder[ti]).cpu().detach().numpy()

            ##################
            ### Optimizing ###
            ##################
            # if ti%2==0:
            self.optimizer.zero_grad()
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()
            self.optimizer.step()
            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti].cpu().detach().numpy(), "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti].cpu().detach().numpy(),"[dB]","timing ", time.time() - t)
            print(ti, "Enc Training :", self.MSE_train_dB_epoch_encoder[ti].cpu().detach().numpy(), "[dB]", "Enc Validation :", self.MSE_cv_dB_epoch_encoder[ti].cpu().detach().numpy(),"[dB]")
            Train_loss_list.append(self.MSE_train_dB_epoch[ti].cpu().detach().numpy())
            Val_loss_list.append(self.MSE_cv_dB_epoch[ti].cpu().detach().numpy())
        #self.print_process(Val_loss_list, Train_loss_list, title)
        print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    # def print_process(self, val_loss_list, train_loss_list, title):
    #     fig = plt.figure(figsize=(10, 7))
    #     fig.add_subplot(1, 1, 1)
    #     plt.plot(train_loss_list, label='train')
    #     plt.plot(val_loss_list, label='val')
    #     plt.title("Loss of {}".format(title))
    #     plt.legend()
    #     plt.savefig(self.Learning_process_folderName +'Learning_curve.jpeg')

    def NNTest(self, n_Test, test_input, test_target,prior_flag,d):
        length_seq = test_input.shape[1]
        self.MSE_test_linear_arr = torch.empty([n_Test])
        self.MSE_test_linear_arr_encoder = torch.empty([n_Test])
        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')
        self.model.eval()
        torch.no_grad()
        x_out_test_all = torch.empty(n_Test, self.ssModel.m, length_seq)
        encoder_test_all = torch.empty(n_Test, d, length_seq)
        Latent_KalmanNet_time = []
        for j in range(0, n_Test): #running on each sample (trajectory)
            start = time.time()
            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)
            y_mdl_tst = test_input[j, :, :] #taking the j observation sample (trajectory)
            x_out, z_encoder_output = self.get_seq_knet_output(y_mdl_tst,test_target[j,:,:],prior_flag)
            x_out_test_all[j,:,:] = x_out
            encoder_test_all[j,:,:] = z_encoder_output
            if self.dataset_name == "Pendulum":
                self.MSE_test_linear_arr[j] = loss_fn(x_out[0,:], test_target[j, 0, :]).detach()
                self.MSE_test_linear_arr_encoder[j] = loss_fn(z_encoder_output, test_target[j, 0, :]).detach()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out, test_target[j, :, :]).detach()
                self.MSE_test_linear_arr_encoder[j]= loss_fn(z_encoder_output, test_target[j, :, :]).detach()
            print(j)
            Latent_KalmanNet_time.append(time.time() - start)
        #print("Latent KalmanNet average time for trajectory is {}".format(mean(Latent_KalmanNet_time)))
        ####### Latent KalmanNet #####################
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        loss_var_set = 0
        for loss_traj in self.MSE_test_linear_arr:
            loss_var_set += (self.MSE_test_linear_avg - loss_traj)*(self.MSE_test_linear_avg - loss_traj)
        loss_var_set = np.sqrt(loss_var_set) / np.sqrt(len(test_input))

        print("Latent KalmanNet Test loss: {} dB with variance {} dB".format(self.MSE_test_dB_avg, 10 * torch.log10(self.MSE_test_linear_avg  + loss_var_set) - self.MSE_test_dB_avg))

        # Standard deviation
        #self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        #self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)
        # Print MSE Cross Validation
        #str = self.sinerio + "-" + "MSE Test:"
        #print(str, self.MSE_test_dB_avg, "[dB]")
        #str = self.sinerio + "-" + "STD Test:"
        #print(str, self.MSE_test_dB_std, "[dB]")


        # Print Run Time
        #print("Inference Time:", infer_time)


        # histogram of KFLS
        #plt.hist(self.MSE_test_linear_arr, 10)
        #plt.show()

        ####### Encoder #####################
        # Average
        self.MSE_test_linear_avg_encoder = torch.mean(self.MSE_test_linear_arr_encoder)
        self.MSE_test_dB_avg_encoder = 10 * torch.log10(self.MSE_test_linear_avg_encoder)

        # Standard deviation
        loss_var_encoder_set = 0
        for loss_traj in self.MSE_test_linear_arr_encoder:
            loss_var_encoder_set += (self.MSE_test_linear_avg_encoder - loss_traj) * (self.MSE_test_linear_avg_encoder - loss_traj)
        loss_var_encoder_set = np.sqrt(loss_var_encoder_set) / np.sqrt(len(test_input))

        print("Only Encoder Test loss: {} dB with variance {} dB".format(self.MSE_test_dB_avg_encoder, 10 * torch.log10(
            self.MSE_test_linear_avg_encoder + loss_var_encoder_set) - self.MSE_test_dB_avg_encoder))

        # Standard deviation
        #self.MSE_test_dB_std_encoder = torch.std(self.MSE_test_linear_arr_encoder, unbiased=True)
        #self.MSE_test_dB_std_encoder = 10 * torch.log10(self.MSE_test_dB_std_encoder)
        # Print MSE Cross Validation
        #str = "Only Encoder" + "-" + "MSE Test:"
        #print(str, self.MSE_test_dB_avg_encoder, "[dB]")
        #str = "Only Encoder" + "-" + "STD Test:"
        #print(str, self.MSE_test_dB_std_encoder, "[dB]")

        # histogram of encoder
        #plt.hist(self.MSE_test_linear_arr_encoder, 10)
        #plt.show()

        return [encoder_test_all, x_out_test_all]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.sinerio)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg, self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)