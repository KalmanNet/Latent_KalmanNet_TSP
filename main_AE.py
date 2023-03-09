###### main_AE ################
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Extended_data_visual import DataLoader_GPU
import numpy as np
import math
import random

def define_dev():
  if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
  else:
    dev = torch.device("cpu")
    print("Running on the CPU")
  return dev

def train_with_prior(path_enc, encoder, train_input, train_target, cv_input, cv_target, num_epochs,batch_size):
    encoder.train()
    encoder.float()
    num_trajectories = train_input.shape[0]
    best_val_loss=1000000
    ############ Learning Configurations ###############
    loss_fn = torch.nn.MSELoss(reduction='mean')
    params_to_optimize = [{'params': encoder.parameters()}]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for k in range(batch_size):
            n_chosen = random.randint(0, train_input.shape[0] - 1)
            trajectory_target = train_target[n_chosen][0, :].unsqueeze(0)
            trajectory_obs = train_input[n_chosen]
            if dataset_name == "Pendulum":
                state_prev = torch.stack((torch.ones(1) * 90 * np.pi / 180, torch.zeros(1)), 0)
            else:
                state_prev = torch.ones(test_target.shape[1], 1)
            x_out = torch.empty(trajectory_target.shape[0], test_target.shape[2])
            for t in range(trajectory_obs.shape[0]):  # running over all time steps
                obs = torch.from_numpy(trajectory_obs[t, :, :]).unsqueeze(0).unsqueeze(0)
                if dataset_name == "Lorenz":
                    prior = torch.matmul(f_function(state_prev.float()), state_prev.float()).transpose(0, 1)
                else:
                    prior = f_function(state_prev).clone()
                    state_prev = prior.clone()
                encoder = encoder.double()
                encoded_data = encoder(obs.double(), prior[:, 0].unsqueeze(0).double()).clone()
                x_out[:, t] = encoded_data
                state_prev[:, 0] = encoded_data
                state_prev = state_prev.transpose(0, 1)
            loss_trajectory = loss_fn(x_out.double(), trajectory_target)
            train_loss = train_loss + loss_trajectory
        optimizer.zero_grad()
        train_loss_mean = train_loss / num_trajectories
        train_loss_mean.backward()
        optimizer.step()
        epoch_val_loss = inference_with_prior(encoder, cv_input, cv_target, dataset_name,f_function)
        print('epoch: {} train loss: {} dB val loss: {} dB'.format(epoch, train_loss_mean,epoch_val_loss))
        if epoch_val_loss < best_val_loss:
            torch.save(encoder.state_dict(), path_enc)
            best_val_loss = epoch_val_loss
    return encoder

def train(path_enc,encoder, train_loader, val_loader, num_epochs, batch_size, flag_prior):
    encoder.train()
    encoder.float()
    best_val_loss=1000000
    ############ Learning Configurations ###############
    loss_fn = torch.nn.MSELoss()
    params_to_optimize = [{'params': encoder.parameters()}]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    train_loss_epoch=[]
    val_loss_epoch = []
    for epoch in range(num_epochs):
        train_loss = []
        for k, batch in enumerate(train_loader):
            if (batch[0].shape[0] == batch_size):  # taking only full batch to learn from
                image_batch = batch[0].float()
                if dataset_name =="Pendulum":
                    states_with_noise_batch = batch[1][:,0].unsqueeze(1)
                    targets_batch = batch[2][:,0].unsqueeze(1)
                else:
                    states_with_noise_batch = batch[1]
                    targets_batch = batch[2]
                encoder = encoder.float()
                if flag_prior:
                    encoded_data = encoder(image_batch,states_with_noise_batch)
                else:
                    encoded_data = encoder(image_batch)
                loss = loss_fn(encoded_data, targets_batch)
                if torch.isnan(loss):
                    print("we have nan")
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                train_loss.append(loss.detach().numpy())
        train_loss_epoch.append(10 * math.log10(np.mean(train_loss)))
        epoch_val_loss = test_epoch(encoder, val_loader, loss_fn, batch_size, flag_prior, dataset_name)
        val_loss_epoch.append(10 * math.log10(epoch_val_loss))
        print('epoch: {} train loss: {} dB val loss: {} dB'.format(epoch, train_loss_epoch[epoch],val_loss_epoch[epoch]))
        if val_loss_epoch[epoch] < best_val_loss:
            torch.save(encoder.state_dict(), path_enc)
    return encoder, val_loss_epoch, train_loss_epoch

# def get_encoder_output(test_input, model_encoder_trained):
#     x_out_test_all = torch.empty(test_input.shape[0], d, test_input.shape[2])
#     for j in range(0, test_input.shape[0]):
#         objervation_sample = test_input[j, :, :]
#         encoder_sample_output = torch.empty(d, test_input.shape[2])
#         for t in range(0, test_input.shape[2]):
#             AE_input = objervation_sample[:, t].reshape(1, 1, y_size, y_size)
#             encoder_sample_output[:,t] = model_encoder_trained(AE_input).squeeze()
#         x_out_test_all[j,:,:]=encoder_sample_output
#     return x_out_test_all

def test_epoch(encoder, val_loader, loss_fn, batch_size, flag_prior,dataset_name):
    #encoder_test = torch.empty(100, 3, 2000)
    encoder.eval()
    with torch.no_grad(): # No need to track the gradients
        val_loss = []
        for k, batch in enumerate(val_loader):
            if (batch[0].shape[0] == batch_size):  # taking only full batch to learn from
                image_batch = batch[0]
                if dataset_name =="Pendulum":
                    states_with_noise_batch = batch[1][:,0].unsqueeze(1)
                    targets_batch = batch[2][:,0].unsqueeze(1)
                else:
                    states_with_noise_batch = batch[1]
                    targets_batch = batch[2]
                encoder = encoder.float()
                if flag_prior:
                    encoded_data = encoder(image_batch,states_with_noise_batch)
                else:
                    encoded_data = encoder(image_batch)
                loss = loss_fn(encoded_data, targets_batch)
                if torch.isnan(loss):
                    print("we have nan")
                val_loss.append(loss.numpy())
    return np.mean(val_loss)

def inference_with_prior(encoder, test_input, test_target, dataset_name,f_function):
    encoder.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad(): # No need to track the gradients
        test_loss = []
        for k,trajectory_obs in enumerate(test_input):
            if dataset_name == "Pendulum":
                trajectory_target = test_target[k][0, :].unsqueeze(0)
                state_prev = torch.stack((torch.ones(1)* 90* np.pi / 180,torch.zeros(1)), 0)
            else:
                trajectory_target = test_target[k]
                state_prev = torch.ones(test_target.shape[1], 1)
            x_out = torch.empty(trajectory_target.shape[0], test_target.shape[2])
            for t in range(trajectory_obs.shape[0]): #running over all time steps
                obs = torch.from_numpy(trajectory_obs[t,:,:]).unsqueeze(0).unsqueeze(0)
                encoder = encoder.double()
                if dataset_name == "Lorenz":
                    prior = torch.matmul(f_function(state_prev.float()), state_prev.float()).transpose(0, 1)
                    encoded_data = encoder(obs.double(),prior.double())
                    state_prev = encoded_data
                else:
                    prior = f_function(state_prev)
                    state_prev = prior
                    encoded_data = encoder(obs.double(), prior[:, 0].unsqueeze(0).double())
                    state_prev[:, 0] = encoded_data.squeeze(0)
                x_out[:, t] = encoded_data
                state_prev = state_prev.transpose(0,1)
            test_loss.append(loss_fn(x_out, trajectory_target).detach().numpy()) ## collecting mu_traj
        loss_mu_set = np.mean(test_loss)
        loss_var_set = 0
        for loss_traj in test_loss:
            loss_var_set+=np.power(loss_mu_set-loss_traj,2)
        loss_var_set = np.sqrt(loss_var_set)/np.sqrt(len(test_input))
        return loss_mu_set, loss_var_set

class Encoder_conv(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(288, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x):
        #x=x.type(torch.DoubleTensor)
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

#class Encoder_conv(nn.Module):
#    def __init__(self, encoded_space_dim):
#        super(Encoder_conv, self).__init__()
#       ### Convolutional section
#       #self.encoder_conv = nn.Sequential(
#        self.conv2d1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
#        self.relu = nn.ReLU(True)
#        self.drop = nn.Dropout(0.5)
#        self.conv2d2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
#        self.conv2d3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
#        self.batchNorm1 = nn.BatchNorm2d(8)
#        self.batchNorm2 = nn.BatchNorm2d(16)
#        self.batchNorm3 = nn.BatchNorm2d(32)
#       ### Flatten layer
#        self.flatten = nn.Flatten(start_dim=1)
#       ### Linear section
#       self.fc1 = nn.Linear(512, 32)
#        self.fc2 = nn.Linear(32, encoded_space_dim)

#    def forward(self, x):
#        x = self.relu(self.conv2d1(x))
#        x = self.batchNorm1(x)
#        x = self.relu(self.conv2d2(x))
#        x = self.batchNorm2(x)
#        x = self.relu(self.conv2d3(x))
#        x = self.batchNorm3(x)
#        x = self.flatten(x)
#        x = self.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x


class Encoder_conv_with_prior(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv_with_prior, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.prior_fc = nn.Sequential(
            nn.Linear(encoded_space_dim, 8),
            #nn.ReLU(True),
            nn.Dropout(0.5))
        self.encoder_lin = nn.Sequential(
            nn.Linear(296, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x, prior):
        #x=x.type(torch.DoubleTensor)
        if torch.isnan(x).any():
            itay=29
        x = self.encoder_conv(x)
        x = self.flatten(x)
        prior = self.prior_fc(prior)
        comb = torch.cat((prior, x), 1)
        out = self.encoder_lin(comb)
        return out

def check_learning_process(img_batch,recon_batch,epoch, name):
    y_nump = img_batch[32].reshape(28,28).detach().numpy()
    #reshape(24,24).detach().numpy().squeeze()
    y_recon_nump = recon_batch[32].reshape(28,28).detach().numpy()
    #recon_batch[32].reshape(24,24).detach().numpy().squeeze()
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_nump)
    #cmap = 'gray'
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_recon_nump)
    #, cmap='gray'
    plt.axis('off')
    plt.title("reconstruct")
    fig.savefig('AE Process/{} Process at epoch {}.PNG'.format(name, epoch))

def print_process(val_loss_list, train_loss_list,r):
    configuration='only encoder'
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title("Loss of "+configuration)
    plt.xlabel("epochs")
    plt.ylabel("MSE [dB]")
    plt.legend()
    plt.savefig(r".\AE Process\Learning_process encoder with prior r={} prior r={}.jpg".format(r,prior_r))

def create_dataset_loader(imgs_np, states_np_with_noise, states_np, batch_size):
    # create imgs list
    data_list = []
    for k in range(imgs_np.shape[0]):
        sample = imgs_np[k]#test set
        seq_length = sample.shape[1]
        for t in range(seq_length):
            img = sample[:, t].reshape((1, 28, 28))
            data_list.append(img)

    # create states with noise list
    data_state_noise_list = []
    for k in range(states_np_with_noise.shape[0]):
        sample = states_np_with_noise[k]#test set
        for t in range(seq_length):
            state_with_noise = sample[:, t]
            data_state_noise_list.append(state_with_noise)

    # create target list ##
    chosen_targets_np= states_np
    targets_list = []
    for k in range(chosen_targets_np.shape[0]):
        sample = chosen_targets_np[k]
        for t in range(seq_length):
            target = sample[:,t]
            targets_list.append(target)

    dataset=[]
    for k in range(len(data_list)):
        dataset.append((data_list[k],data_state_noise_list[k],targets_list[k]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return loader

def add_noise(states_np_train,r):
    stats_np_with_noise = np.empty_like(states_np_train)
    for k in range(states_np_train.shape[0]): #runing over each trajectory
        for i in range(states_np_train.shape[2]):#running over each time step
            added_noise = np.random.normal(0, r, states_np_train.shape[1])
            stats_np_with_noise[k,:,i] = states_np_train[k,:,i]+added_noise
    return stats_np_with_noise

def create_dataset_loader_Pendulum(observations, stats_with_noise, states, batch_size,warm_start_flag):
    # create observations list
    observations_list = []
    observations = torch.from_numpy(observations).float()
    for k in range(observations.shape[0]):
        sample = observations[k]  # test set
        seq_length = sample.shape[0]
        for t in range(seq_length):
            if warm_start_flag:
                img = sample[:,t]
            else:
                img = sample[t, :, :].reshape((1, 28, 28))
            observations_list.append(img)

    # create states list
    states_list = []
    states = torch.from_numpy(states).float()
    for k in range(states.shape[0]):
        sample = states[k]  # test set
        for t in range(seq_length):
            state = sample[:, t]
            states_list.append(state)

    # create states with noise list
    state_noise_list = []
    stats_with_noise = torch.from_numpy(stats_with_noise).float()
    for k in range(stats_with_noise.shape[0]):
        sample = stats_with_noise[k]#test set
        for t in range(seq_length):
            state_with_noise = sample[:, t]
            state_noise_list.append(state_with_noise)

    dataset = []
    for k in range(len(observations_list)):
        dataset.append((observations_list[k], state_noise_list[k], states_list[k]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return loader

def initialize_data_AE_Pendulum(path_for_states,path_for_observations,batch_size, prior_r, real_r, dev, warm_start_flag):
    States = np.load(path_for_states)
    Observations = np.load(path_for_observations)

    train_loader, train_input, train_target = initialize_data_AE_Pendulum_per_dataset(States, Observations,batch_size, prior_r, real_r,warm_start_flag, 'training_set')
    val_loader, cv_input, cv_target = initialize_data_AE_Pendulum_per_dataset(States, Observations,batch_size, prior_r, real_r,warm_start_flag, 'validation_set')
    test_loader, test_input, test_target = initialize_data_AE_Pendulum_per_dataset(States, Observations,batch_size,prior_r, real_r,warm_start_flag, 'test_set')

    test_target = torch.from_numpy(test_target).to(dev)
    train_target = torch.from_numpy(train_target).to(dev)
    cv_target = torch.from_numpy(cv_target).to(dev)
    return train_loader, val_loader, test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target

def initialize_data_AE_Pendulum_per_dataset(States, Observations, batch_size, prior_r, prob_r, warm_start_flag, type):
    input = Observations[type].astype(float)
    input = sp_noise(input,prob_r)
    #np.sqrt(real_r) * np.random.standard_normal(input.shape)
    target = States[type]
    stats_with_noise = target + np.sqrt(prior_r) * np.random.standard_normal(target.shape)
    loader = create_dataset_loader_Pendulum(input,stats_with_noise,target,batch_size,warm_start_flag)
    return loader, input, target

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    probs = np.random.random(output.shape)
    output[probs < (prob / 2)] = 0
    output[probs > 1 - (prob / 2)] = 1
    return output

def initialize_data_AE_Lorenz(path_for_data, batch_size,prior_r, y_size):
    ### load data ###
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(path_for_data)
    ############ create datasets ####################
    imgs_np_train=train_input.cpu().detach().numpy()
    states_np_train=train_target.cpu().detach().numpy()
    stats_np_with_noise_train=add_noise(states_np_train,prior_r)
    train_loader = create_dataset_loader(imgs_np_train, stats_np_with_noise_train, states_np_train, batch_size)

    imgs_np_val=cv_input.cpu().detach().numpy()
    states_np_val=cv_target.cpu().detach().numpy()
    stats_np_with_noise_val = add_noise(states_np_val, prior_r)
    val_loader = create_dataset_loader(imgs_np_val, stats_np_with_noise_val, states_np_val, batch_size)

    imgs_np_test=test_input.cpu().detach().numpy()
    states_np_test=test_target.cpu().detach().numpy()
    stats_np_with_noise_test = add_noise(states_np_test, prior_r)
    test_loader = create_dataset_loader(imgs_np_test, stats_np_with_noise_test, states_np_test, batch_size)

    test_input = reformat(test_input, y_size).detach().numpy()
    train_input = reformat(train_input, y_size).detach().numpy()
    cv_input = reformat(cv_input, y_size).detach().numpy()
    return train_loader,val_loader,test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target

def reformat(data_input,y_size):
    data_input = data_input.transpose(1, 2)
    data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], y_size, y_size)
    return data_input

# r_2_values = [10,4,2,1]
# one_div_r_2_values = [1/r_2 for r_2 in r_2_values]
# x_axis = 10*np.log10(one_div_r_2_values)
# results_encoder = [-6.44,-8.3,-11.08,-15.47]
# results_encoder_with_prior = [-6.9,-9.11,-11.6,-15.41]
# fig, ax = plt.subplots()
# ax.plot(x_axis,results_encoder, marker='o', label='Encoder', color='red' )
# ax.plot(x_axis,results_encoder_with_prior, marker='o', label='Encoder with prior', color='yellow')
# plt.title(r'MSE vs $\frac{1}{r^2}$ over Baseline')
# plt.xlabel(r'$\frac{1}{r^2}$ [dB]')
# plt.ylabel('MSE [dB]')
# plt.legend()
# plt.grid(True)
# plt.show()

if __name__ == '__main__':
    ## seed
    num =0
    torch.manual_seed(num)
    random.seed(num)
    ## flags
    dataset_name = "Lorenz"
    sinerio = "Baseline"    # "Baseline", "Decimation"
    flag_prior = True
    flag_train = False
    flag_load_model = True
    dev = define_dev()
    warm_start_flag = False

    if dataset_name == "Pendulum":
        from model_Pendulum import *
    else:
        from model_Lorenz import *

    ## Hypper parameters
    lr = 0.0005
    wd = 0.01
    batch_size = 512
    num_epochs = 30
    prior_r2=1

    for r2 in [0.1,0.01,0]: #,0.09 0.04 0.009 0.001
        if r2 == 0.5:
            prior_r2=4 # "Baseline", "Decimation", "Test_Long_Trajectories"
        ############ data sinerio #################
        path_for_states = rf'.\Simulations\{dataset_name}\states_q2_{real_q2}_{sinerio}.npz'
        path_for_observations = rf'.\Simulations\{dataset_name}\observations_q2_{real_q2}_{sinerio}.npz'
        train_loader, valid_loader, test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target = initialize_data_AE_Pendulum(path_for_states,path_for_observations,batch_size, prior_r2, r2, dev, warm_start_flag)

        ### choosing model
        if flag_prior:
            encoder = Encoder_conv_with_prior(encoded_space_dim=d).double()
            path_enc = r'.\Encoder\{}\{}\{}_Only_encoder_r={}_prior={}.pt'.format(dataset_name, sinerio+ "_with_prior", dataset_name, r2, prior_r2)
            print("### Dataset: {} with r: {}, Sinerio: {} {} ###".format(dataset_name, r2, sinerio+ "_with_prior",prior_r2))
        else:
            encoder = Encoder_conv(encoded_space_dim=d).double()
            path_enc = r'.\Encoder\{}\{}\{}_Only_encoder_r={}.pt'.format(dataset_name, sinerio, dataset_name, r2)
            print("### Dataset: {} with r: {}, Sinerio: {} ###".format(dataset_name, r2, sinerio))
        print('### Parameters lr: {} wd: {} ###'.format(lr, wd))

        ## load model
        if flag_load_model:
            encoder.load_state_dict(torch.load(path_enc), strict=False)

        ### train model
        if flag_train:
            encoder, val_loss_epoch, train_loss_epoch = train(path_enc,encoder, train_loader,valid_loader, num_epochs, batch_size, flag_prior)
            #print_process(val_loss_epoch, train_loss_epoch,r)

        ### test model
        if flag_prior:
            test_loss_mu, test_loss_var = inference_with_prior(encoder, test_input, test_target, dataset_name, f_function)
            print("Test loss with prior {} is: {} dB with variance {} dB".format(prior_r2, 10 * math.log10(
            test_loss_mu), 10 * math.log10(test_loss_mu + test_loss_var) - 10 * math.log10(test_loss_mu)))
        else:
            test_loss_mu = test_epoch(encoder, test_loader, torch.nn.MSELoss(), batch_size, flag_prior,dataset_name)
            print("Test loss is: {} dB".format(10 * math.log10(test_loss_mu)))



