## Extended_data_visual ##
import torch
import math
import os
from model_Lorenz import m
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    #print("Running on the CPU")

#######################
### Size of DataSet ###
#######################

#################
## Design #10 ###
#################
F10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# H_matrix_5 = torch.tensor([[1.0, 0.0],
#                            [0.0, 1.0],
#                            [0.5, 0.5],
#                            [0.75, 0.25],
#                            [0.25, 0.75]])
#
# H_matrix_2 = torch.tensor([[1.0, 0.0],
#                            [0.0, 1.0]])

# b_2 = torch.tensor([[0.0],
#                     [0.0]])

# b_5 = torch.tensor([[0.0],
#                     [0.0],
#                     [0.0],
#                     [0.0],
#                     [0.0]])


############
## 2 x 2 ###
############
# m = 2
# n = 2
# F = F10[0:m, 0:m]
# H = torch.eye(2)
m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(dev)

#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# F = F10[0:m, 0:m]
# H = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(dev)
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# F = F10[0:m, 0:m]
# H = H10
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

# Inaccurate model knowledge based on matrix rotation
alpha_degree = 10
rotate_alpha = torch.tensor([alpha_degree / 180 * torch.pi]).to(dev)
cos_alpha = torch.cos(rotate_alpha)
sin_alpha = torch.sin(rotate_alpha)
rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                              [sin_alpha, cos_alpha]]).to(dev)
# print(rotate_matrix)
#F_rotated = torch.mm(F, rotate_matrix)  # inaccurate process model
#H_rotated = torch.mm(H, rotate_matrix)  # inaccurate observation model


def DataGen_True(SysModel_data, fileName, T):
    SysModel_data.GenerateBatch(1, T, randomInit=False)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},fileName)
    torch.save([test_input, test_target], fileName)


def DataGen(SysModel_data, dataset_name, sinerio, T, T_test, N_E, N_CV, N_T, randomInit=False):
    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    np.savez(rf"Simulations/{dataset_name}/observations_q2_{SysModel_data.real_q2}_{sinerio}.npz",
             training_set=training_input.numpy(),validation_set=cv_input.numpy(),test_set=test_input.numpy())

    np.savez(rf"Simulations/{dataset_name}/states_q2_{SysModel_data.real_q2}_{sinerio}.npz",
             training_set=training_target.numpy(), validation_set=cv_target.numpy(), test_set=test_target.numpy())

    #torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], './Simulations/lorenz_T=200_decimated_q=0.1_r={}.pt'.format(SysModel_data.real_r))
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]


def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(
        torch.load(fileName), pin_memory=False)
    training_input = training_input.squeeze().to(dev)
    training_target = training_target.squeeze().to(dev)
    cv_input = cv_input.squeeze().to(dev)
    cv_target = cv_target.squeeze().to(dev)
    test_input = test_input.squeeze().to(dev)
    test_target = test_target.squeeze().to(dev)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]


def DecimateData(all_tensors, t_gen, t_mod, offset=0):
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod / t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:, (0 + offset)::ratio]
        if (i == 0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1, all_tensors.size()[1], -1)
        else:
            all_tensors_out = torch.cat([all_tensors_out, tensor], dim=0)
        i += 1

    return all_tensors_out


def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process, h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples) * [decimated_process])
    noise_free_obs = torch.cat(int(N_examples) * [noise_free_obs])

    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]


def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i, :, t] = h(sequence[:, t])
    i = i + 1

    return sequences_out


def Short_Traj_Split(data_target, data_input, T):
    data_target = list(torch.split(data_target, T, 2))
    data_input = list(torch.split(data_input, T, 2))
    data_target.pop()
    data_input.pop()
    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
    return [data_target, data_input]