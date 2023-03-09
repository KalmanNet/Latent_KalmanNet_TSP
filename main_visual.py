## main_visual ##
from EKFTest_visual import EKFTest
from Pipeline_KF_visual_gpu import Pipeline_KF
from KalmanNet_nn_visual_new_architecture_gpu import KalmanNetLatentNN, in_mult, out_mult
from config import *
from main_AE import initialize_data_AE_Pendulum, test_epoch, inference_with_prior
import matplotlib.pyplot as plt
import time

############## checking only encoder loss (not depand on seed) ####################################
train_loader,val_loader,test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target = initialize_data_AE_Pendulum(path_for_states, path_for_observations, batch_size, prior_r2, real_r2, dev, warm_start_flag)

#if prior_flag:
#    test_loss_mu, test_loss_var = inference_with_prior(model_encoder_trained, test_input, test_target, dataset_name, f_function)
#    print("4. Test loss of encoder with prior {} is: {} dB with variance {} dB".format(prior_r2, 10 * math.log10(test_loss_mu),10 * math.log10(test_loss_mu+test_loss_var)-10 * math.log10(test_loss_mu)))
#else:
#    test_loss = test_epoch(model_encoder_trained, test_loader, torch.nn.MSELoss(), batch_size, prior_flag,dataset_name)
#    print("4. Test loss of encoder is: {} dB".format(10 * math.log10(test_loss)))
####################################################################################################

### Evaluate Extended Kalman Filter ###############
if Evaluate_EKF_flag:
    #for r in given_r_options:
    #for q in given_q_options:
    r = given_r[dict.get(real_r2)]
    q = given_q[dict.get(real_r2)]
    print('given q = {} given r = {} '.format(q, r))
    sys_model.givenQ = q * q * torch.eye(m)
    sys_model.givenR = r * r * torch.eye(m)
    print("5. Evaluate Extended Kalman Filter {} {} dataset with real_q = {} and prob_r = {}".format(dataset_name,sinerio,real_q2, real_r2))
    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg_with_encoder, EKF_out_with_encoder, GT_test] = EKFTest(sys_model, test_input, test_target, dataset_name, sinerio,prior_flag, model_encoder_trained,d)
###################################################################################################################

################## KNet Pipeline #########################################################
print("5. Created KNetLatent pipeline instance")
Kalman_Latent_Pipeline = Pipeline_KF(strTime, path_KNetLatent_trained, dataset_name, sinerio, fix_encoder_flag, sys_model,d,warm_start_flag)
################## KGain model
if load_KNetLatent_trained:
    Kalman_Latent_Pipeline.model = torch.load(path_KNetLatent_trained).double()
    print("6. Loaded weights for KNetLatent")
else:
    print("6. Create KNetLatent model")
    KNatLatent_model = KalmanNetLatentNN(dataset_name).double()
    KNatLatent_model.Build(sys_model, model_encoder_trained,fix_encoder_flag,d)
    Kalman_Latent_Pipeline.setLearnedModel(KNatLatent_model.double())
############################################################################################

################## training ################################################################
if flag_Train:
    if fix_encoder_flag:
        print("7. Start KNetLatent pipeline with fix encoder, training over training set")
    else:
        print("7. Start KNetLatent pipeline with learnable encoder, training over training set")
    Kalman_Latent_Pipeline.setTrainingParams(n_Epochs=epoches, n_Batch=batch_size, learningRate=lr_kalman, weightDecay=wd_kalman)
    title="LR: {} Weight Decay: {} model complexity in_mult = {} out_mult = {} evolution function J={} prior r={}".format(lr_kalman,wd_kalman,in_mult, out_mult,J,prior_r2 )
    print(title)
    #print_weights(Kalman_Latent_Pipeline, 'before')
    Kalman_Latent_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target, title, prior_flag)
    #print_weights(Kalman_Latent_Pipeline, 'after')

##################### Compare models on Test Set ##########################################
if fix_encoder_flag:
    print("start  KNet pipeline with fix encoder, inference over test set")
else:
    print("start  KNet pipeline with learnable encoder, inference over test set")
[encoder_test, KNet_test] = Kalman_Latent_Pipeline.NNTest(N_T, test_input, test_target, prior_flag,d)
pytorch_total_params = sum(p.numel() for p in Kalman_Latent_Pipeline.model.parameters() if p.requires_grad)
#print("Knet Pipeline include {} trainable parameters".format(pytorch_total_params))

######################## convert to numpy for rkn code ##############################
# x=test_target[68]
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot(x[0,:], x[1,:], x[2,:])
# plt.draw()
# plt.show()

# train_input_reshaped=train_input.transpose(1,2).reshape((1000,200,28,28,1)).numpy()
# test_input_reshaped=test_input.transpose(1,2).reshape((100,200,28,28,1)).numpy()
# cv_input_reshaped=cv_input.transpose(1,2).reshape((100,200,28,28,1)).numpy()
# train_target_reshaped=train_target.transpose(1,2).numpy()
# test_target_reshaped=test_target.transpose(1,2).numpy()
# cv_target_reshaped=cv_target.transpose(1,2).numpy()
# print("trainset size: x {} y {}".format(train_target_reshaped.shape,train_input_reshaped.shape))
# print("cvset size: x {} y {}".format(cv_target_reshaped.shape, cv_input_reshaped.shape))
# print("testset size: x {} y {}".format(test_target_reshaped.shape, test_input_reshaped.shape))
# np.save('./Simulations/new_lorenz_for_rkn_T=200_q_{}_r_{}'.format(real_q,real_r), [train_input_reshaped, train_target_reshaped, cv_input_reshaped, cv_target_reshaped, test_input_reshaped, test_target_reshaped])
# [train_input_reshaped, train_target_reshaped, cv_input_reshaped, cv_target_reshaped, test_input_reshaped, test_target_reshaped]=np.load('./Simulations/new_lorenz_for_rkn_T=200_q_{}_r_{}.npy'.format(real_q,real_r),allow_pickle=True)
# Itay=29

################################ Plot Trajectories #############################################################
if dataset_name == "Pendulum":
    ## Encoder
    r_values = [0.01, 0.1, 0.2, 0.4, 0.9]
    predictions1 = []
    for i, r in enumerate(r_values):
        model_encoder_trained = Encoder_conv(d).double()
        path_enc = r'./Encoder/{}/{}/{}_Only_encoder_r={}.pt'.format(dataset_name, 'Baseline', dataset_name,r)
        model_encoder_trained.load_state_dict(torch.load(path_enc),strict=False)
        train_loader,val_loader,test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target = initialize_data_AE_Pendulum(path_for_states,path_for_observations,batch_size, prior_r2, r, dev, warm_start_flag)
        loss = test_epoch(model_encoder_trained, test_loader, torch.nn.MSELoss(reduction='mean'), batch_size, False, dataset_name)
        print("Encoder loss p_r={}: {}".format(r,10 * math.log10(loss)))
        num_trj = 90
        target = test_target[num_trj, 0, :]
        input = test_input[num_trj, :, :, :]
        Encoder_output = []
        input = test_input[num_trj,:,:,:]
        for k in range(input.shape[0]):
            obs = torch.from_numpy(input[k]).unsqueeze(0).unsqueeze(0)
            Encoder_output.append(model_encoder_trained(obs.float()).detach().numpy()[0])
        predictions1.append(Encoder_output)

    ## Encoder with prior
    r_values = [0.01,0.1,0.2,0.4,0.9]
    predictions = []
    for i,r in enumerate(r_values):
        model_encoder_with_prior = Encoder_conv_with_prior(1).double()
        path_enc = r'./Encoder/{}/{}/{}_Only_encoder_r={}_prior={}.pt'.format(dataset_name, 'Baseline_with_prior', dataset_name, r ,prior_r2)
        model_encoder_with_prior.load_state_dict(torch.load(path_enc),strict=False)
        state_prev = torch.stack((torch.ones(1)* 90* np.pi / 180,torch.zeros(1)), 0)
        train_loader, val_loader, test_loader, train_input, train_target, cv_input, cv_target, test_input, test_target = initialize_data_AE_Pendulum(
            path_for_states, path_for_observations, batch_size, prior_r2, r, dev, warm_start_flag)
        loss = inference_with_prior(model_encoder_with_prior, test_input, test_target, dataset_name,f_function)
        print("Encoder + Prior loss: {}".format(10 * math.log10(loss)))

        #model_encoder_with_prior_from_KNet = Kalman_Latent_Pipeline.model.model_encoder
        #loss = inference_with_prior(model_encoder_with_prior_from_KNet, test_input, test_target, dataset_name,f_function)
        #print("Encoder + Prior from KNetloss: {}".format(10 * math.log10(loss)))
        num_trj = 90
        target = test_target[num_trj, 0, :]
        input = test_input[num_trj, :, :, :]
        Encoder_with_prior_output = []
        for k in range(input.shape[0]):
            obs = torch.from_numpy(input[k]).unsqueeze(0).unsqueeze(0)
            prior = f_function(state_prev)
            output = model_encoder_with_prior(obs,prior[:,0].unsqueeze(0).double()).detach().numpy()[0][0]
            Encoder_with_prior_output.append(output)
            state_prev = prior
            state_prev[:, 0] = output
            state_prev = state_prev.transpose(0, 1)
        predictions.append(Encoder_with_prior_output)
    # # Latent KalmanNet
    # [encoder_test, KNet_test] = Kalman_Latent_Pipeline.NNTest(N_T, test_input, test_target, prior_flag)
    # [y,x] = Kalman_Latent_Pipeline.NNTest(1, np.expand_dims(input, axis=0), np.expand_dims(test_target[num_trj,:,:], axis=0), prior_flag)
    # Latent_kalmanNet_output = [y[:,:,t].detach().numpy()[0][0] for t in range(400)]
    # Encoder_after_alternating_output = [x[:,:,t].detach().numpy()[0][0] for t in range(400)]

    fig, ax = plt.subplots()
    ax.plot(predictions[3], marker='+', label='Encoder after alternating', color='brown')
    ax.plot(predictions[2], marker='o', label='Encoder', color='red')
    ax.plot(predictions[1], marker='x', label='Encoder + Prior', color='orange')
    ax.plot(predictions1[1], marker='s', label='Encoder + Prior + EKF', color='purple')
    ax.plot(predictions1[0], marker='p', label='Latent KalmanNet', color='green')
    ax.plot(target, label='Target', color = 'black')

    plt.title(r'Pendulum Estimated trajectory')
    plt.xlabel(r'Time steps')
    plt.ylabel('Angle')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    fig.savefig('trajectory_design_steps.eps', format='eps', dpi=1200)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(predictions[3][:40], marker='+', label='Encoder after alternating', color='brown')
    ax.plot(predictions[2][:40], marker='o', label='Encoder', color='red')
    ax.plot(predictions[1][:40], marker='x', label='Encoder + Prior', color='orange')
    ax.plot(predictions1[1][:40], marker='s', label='Encoder + Prior + EKF', color='purple')
    ax.plot(predictions1[0][:40], marker='p', label='Latent KalmanNet', color='green')
    ax.plot(target[:40], label='Target', color='black')

    plt.title(r'Zoom In')
    plt.xlabel(r'Time steps')
    plt.ylabel('Angle')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    fig.savefig('trajectory_design_steps_zoom.eps', format='eps', dpi=1200)
    plt.close()

#else:
#    plot_trajectory(Encoder_output_0,target)
#    plot_trajectory(Encoder_output_001,target)
#    plot_trajectory(Encoder_output_01,target)


###########################  plot MSE ##################################
if dataset_name == "Lorenz":
    r_2_values = [0.5,0.1,0.01,0]

    ########### Baseline #################
    results_encoder = [5.8, -0.5, -3.7, -5.6]
    results_encoder_with_prior_r_1_J_5 = [2.58,-2.67,-6.1,-6.84]
    results_encoder_with_prior_r_1_and_EKF_J_5=[0.51, -3, -6.31,-7.16]
    results_KNet_fixed_Encoder_J_5 = [-1.56,-3.27,-6.4,-7.22]
    results_KNet_Latent_J_5 = [-1.91,-4.92,-7.2,-7.94]
    results_Encoder_after_alternating_J_5 = [2,-4.22,-6.73,-7.3]
    results_rkn = [-1,-4.2,-6.9,-7.8]

    plot_MSE(r_2_values,results_encoder,results_encoder_with_prior_r_1_J_5,
             results_encoder_with_prior_r_1_and_EKF_J_5,results_KNet_Latent_J_5,results_rkn,'Baseline')

    ######## Long ###############
    results_rkn_long = [5.7,-0.2,-3.5,-5.8]
    results_encoder_long = [5.4, -0.58, -3.9, -6]
    results_encoder_with_prior_r_1_J_5_long = [1.63,-3.74,-6.72,-7.57]
    results_encoder_with_prior_r_1_and_EKF_J_5_long=[-0.61, -4.34, -7.11,-7.91]
    results_KNet_fixed_Encoder_J_5_long = [-1.93,-4.15,-7.2,-7.85]
    results_KNet_Latent_J_5_long = [-2.4,-5.45,-7.91,-8.54]
    results_Encoder_after_alternating_J_5_long = [1.3,-5.25,-7.61,-8]

    plot_MSE(r_2_values, results_encoder_long, results_encoder_with_prior_r_1_J_5_long,
            results_encoder_with_prior_r_1_and_EKF_J_5_long, results_KNet_Latent_J_5_long, results_rkn_long,'Long_Trajectories')

    ########### Wrong F, J=1 #################
    results_encoder_with_prior_r_1_J_1 = [3.19,-1.1,-5.44,-6.28]
    results_encoder_with_prior_r_1_and_EKF_J_1=[2.8, -1.2, -5.42,-6.29]
    results_KNet_fixed_Encoder_J_1 = [-0.6,-4.3,-6.5,-7.3]
    results_KNet_Latent_J_1 = [-1.61,-4.86,-6.85,-7.53]
    results_Encoder_after_alternating_J_2 = [1.62,-3.39,-6.52,-7.15]

    fig, ax = plt.subplots()
    ax.plot(r_2_values, results_encoder, marker='o', label='Encoder', color='red')
    ax.plot(r_2_values, results_encoder_with_prior_r_1_J_1, marker='x', label='Encoder + Prior', color='orange')
    ax.plot(r_2_values, results_encoder_with_prior_r_1_and_EKF_J_1, marker='s', label='Encoder + Prior + EKF', color='purple')
    ax.plot(r_2_values, results_rkn, marker='^', label='RKN', color='black')
    ax.plot(r_2_values, results_KNet_Latent_J_1, marker='*', label='Latent KalmanNet J=2', color='blue')
    ax.plot(r_2_values, results_KNet_Latent_J_5, marker='p', label='Latent KalmanNet J=5', color='green')
    #plt.title(r"Lorenz MSE over Approximated state evolution function")
    plt.xlabel(r'Probability $p_r$')
    plt.ylabel('MSE [dB]')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    fig.savefig('Approximated_state_evolution_function.eps', format='eps', dpi=1200)
    plt.close()

    ########### Decimation #################
    results_rkn_decimation = [4.3, 0.3, -3.1, -3.5]
    results_encoder = [7.7, 1.78, -1.95, -3.11]
    results_encoder_with_prior_del_002 = [6.1, 0.9, -2.8, -3.2]
    results_encoder_with_prior_and_EKF_del_002 = [5.6, 0.8, -2.83, -3.21]
    results_KNet_fixed_Encoder_del_002 = [6.07, 0.715, -2.82, -3.05]
    results_KNet_Latent_del_002 = [3.3, -0.46, -3.4, -3.6]
    results_Encoder_after_alternating_del_002 = [6.11, 0.53, -3.18, -3.29]

    fig, ax = plt.subplots()
    ax.plot(r_2_values, results_encoder, marker='o', label='Encoder', color='red')
    ax.plot(r_2_values, results_encoder_with_prior_del_002, marker='x', label='Encoder + Prior', color='orange')
    ax.plot(r_2_values, results_encoder_with_prior_and_EKF_del_002, marker='s', label='Encoder + Prior + EKF',color='purple')
    ax.plot(r_2_values, results_rkn_decimation, marker='^', label='RKN', color='black')
    ax.plot(r_2_values, results_KNet_Latent_del_002, marker='p', label='Latent KalmanNet', color='green')
    # plt.title(r"Lorenz MSE over Approximated state evolution function")
    plt.xlabel(r'Probability $p_r$')
    plt.ylabel('MSE [dB]')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    fig.savefig('Decimation.eps', format='eps', dpi=1200)
    plt.close()

else:
    r_2_values = [10,4,2,1]
    db_axis = [10*np.log10(1/r2) for r2 in r_2_values]
    ########### Design steps #################
    results_encoder_after_alternating = [1.6, -0.2, -1.8, -4.5]
    results_encoder = [0, -1.2, -3.1, -4.9]
    results_encoder_with_prior_r_8 = [-4, -5.1, -6.5, -8.28]
    results_encoder_with_prior_r_8_and_EKF = [-4.8, -6.1, -7.3, -9.3]
    results_KNet_Latent = [-8.2, -8.3, -9.8, -11.1]

    fig, ax = plt.subplots()
    ax.plot(db_axis, results_encoder_after_alternating, marker='P', label='Encoder after alternating', color='blue')
    ax.plot(db_axis, results_encoder, marker='o', label='Encoder', color='red')
    ax.plot(db_axis, results_encoder_with_prior_r_8, marker='x', label='Encoder + Prior', color='orange')
    ax.plot(db_axis, results_encoder_with_prior_r_8_and_EKF, marker='s', label='Encoder + Prior + EKF',color='purple')
    ax.plot(db_axis, results_KNet_Latent, marker='p', label='Latent KalmanNet', color='green')
    # plt.title(r"Lorenz MSE over Approximated state evolution function")
    plt.xlabel(r'$\frac{1}{r^2}$ [dB]')
    plt.ylabel('MSE [dB]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    fig.savefig('Pendulum design steps.eps', format='eps', dpi=1200)
    plt.close()

############################ plot predictions of different algorithms ###################
# sample=44
# initial_encoder = torch.empty(3, 2000)
# trajectory = test_input[sample,:,:]
# for y in range(2000):
#     initial_encoder[:,y] = model_encoder_trained(trajectory[:,y].reshape(1,1,28,28))
#
# #initial_encoder = initial_encoder.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# #plt.axis('off')
# plt.grid(True)
# ax.plot(initial_encoder[0, :], initial_encoder[1, :], initial_encoder[2, :], color='red')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(test_target[sample,0,:], test_target[sample,1,:], test_target[sample,2,:], color='black')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# #encoder_test = encoder_test.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(encoder_test[sample,0,:], encoder_test[sample,1,:], encoder_test[sample,2,:], color='purple')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# #KNet_test = KNet_test.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(KNet_test[sample,0,:], KNet_test[sample,1,:], KNet_test[sample,2,:], color='green')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# for t in range(2000):
#     if t%200==0:
#         fig = plt.figure()
#         imgplot = plt.imshow(trajectory[:,t].reshape(28,28))
#         plt.colorbar()
#         plt.show()
##############################################################################################
