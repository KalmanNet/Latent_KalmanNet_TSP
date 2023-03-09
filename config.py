##########  config  ##########
import torch, random, math
from datetime import datetime # getting current time
from main_AE import Encoder_conv, Encoder_conv_with_prior
from config_script import *
from Extended_data_visual import DataGen
from Extended_sysmdl_visual import SystemModel

def get_dataset_size(train_input, cv_input, test_input):
  N_E = train_input.shape[0]
  N_CV = cv_input.shape[0]
  N_T = test_input.shape[0]
  T = train_input.shape[2]
  T_test = test_input.shape[2]
  return N_E, N_CV, N_T, T, T_test

def define_dev():
  if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
  else:
    dev = torch.device("cpu")
    print("Running on the CPU")
  return dev

def print_weights(Knet_pipeline, state):
  print("#### {} training ####".format(state))
  if fix_encoder_flag:
    print("fix encoder and trainable KGain")
  else:
    print("trainable encoder and fix KGain")
  print("## Encoder ##")
  print(Knet_pipeline.model.model_encoder.encoder_conv[0].weight)
  print("## KGain ##")
  print(Knet_pipeline.model.FC1[0].weight)

# def save_sanity_2(train_input, train_target, cv_input, cv_target, test_input, test_target, model_encoder):
#     noisy_states_cv = save_dataset(cv_input, cv_target, model_encoder)
#     noisy_states_test = save_dataset(test_input, test_target, model_encoder)
#     noisy_states_train = save_dataset(train_input, train_target, model_encoder)
#     np.savez(r"Simulations/Pendulum/noisy_states_sanity_2_r2_{}.npz".format(7),
#              training_set=noisy_states_train, validation_set=noisy_states_cv,test_set=noisy_states_test)
#     np.savez(r"Simulations/Pendulum/states_sanity_2_r2_{}.npz".format(7),
#              training_set=train_target.detach().numpy(), validation_set=cv_target.detach().numpy(), test_set=test_target.detach().numpy())

def plot_MSE(x_axis,results_encoder,results_encoder_with_prior_r_1,results_encoder_with_prior_r_1_and_EKF,results_KNet_Latent, results_rkn,case):
    fig, ax = plt.subplots()
    ax.plot(x_axis, results_encoder, marker='o', label='Encoder', color='red')
    ax.plot(x_axis, results_encoder_with_prior_r_1, marker='x', label='Encoder + Prior', color='orange')
    ax.plot(x_axis, results_encoder_with_prior_r_1_and_EKF, marker='s', label='Encoder + Prior + EKF',color='purple')
    ax.plot(x_axis, results_rkn, marker='^', label='RKN', color='black')
    ax.plot(x_axis, results_KNet_Latent, marker='*', label='Latent KalmanNet', color='green')
    #plt.title(r"Lorenz MSE over {}".format(case))
    plt.xlabel(r'Probability $p_r$')
    plt.ylabel('MSE [dB]')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    fig.savefig('{}.eps'.format(case), format='eps', dpi=1200)
    plt.close()

def plot_trajectory(Encoder_output,target):
    x = [est[0] for est in Encoder_output]
    y = [est[1] for est in Encoder_output]
    z = [est[2] for est in Encoder_output]
    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, label='est', color='blue')
    ax[0].plot(target[0,:], label='GT', color='red')
    plt.title('p_r=0')
    #ax[0].title(r'Estimated x in given trajectory')
    ax[1].plot(y, label='est', color='blue')
    ax[1].plot(target[1, :], label='GT', color='red')
    #ax[1].title(r'Estimated y in given trajectory')
    ax[2].plot(z, label='est', color='blue')
    ax[2].plot(target[0, :], label='GT', color='red')
    #ax[2].title(r'Estimated z in given trajectory')
    plt.xlabel(r'Time steps')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def save_dataset(input, target, encoder):
    encoder.eval()
    loss_fn = torch.nn.MSELoss()
    x_out_all = torch.empty(target.shape[0], 1, target.shape[2])
    with torch.no_grad():  # No need to track the gradients
        loss = []
        for k, trajectory_obs in enumerate(input): #running over all trajectories
            print(k)
            trajectory_target = target[k][0, :].unsqueeze(0)
            state_prev = torch.stack((torch.ones(1) * 45 * np.pi / 180, torch.zeros(1)), 0) #[0.785,0]
            x_out = torch.empty(trajectory_target.shape[0], target.shape[2])
            for t in range(trajectory_obs.shape[0]):  # running over all time steps in the given k trajectory
                obs = torch.from_numpy(trajectory_obs[t, :, :]).unsqueeze(0).unsqueeze(0) # the t image y_t
                prior = f_function(state_prev) #x_t|t-1 (2 elements)
                state_prev = prior #x_t|t-1 (2 elements)
                encoder = encoder.double()
                encoded_data = encoder(obs.double(), prior[:, 0].unsqueeze(0).double()) #z_t# (1 elements)
                x_out[:, t] = encoded_data
                state_prev[:, 0] = encoded_data
                state_prev = state_prev.transpose(0, 1)
            loss.append(loss_fn(x_out, trajectory_target).detach().numpy())
            x_out_all[k,:,:] = x_out
        print("loss: {}".format((10 * math.log10(np.mean(loss)))))
        return(x_out_all)

### Get Time ####################################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
dev = define_dev() # define if working on CPU or GPU with dev
##################################################

############# set seed  ###################################################################
num = 2
torch.manual_seed(num)
random.seed(num)
print("1. Set seed to be {}".format(num))
############################################################################################

### Design SS Model #####################################################################
if dataset_name == "Pendulum":
    from model_Pendulum import *
else:
    from model_Lorenz import *
sys_model = SystemModel(f_function, 0, real_q2, m, h_function, H, 0, real_r2, n, T, T_test, dataset_name)
sys_model.InitSequence(m1x_0, m2x_0)
print("2. Created system model")
############################################################################################

### Data ########################################
if dataset_name == "Pendulum":
    if data_gen_flag:
        print("3. Start data generation {} sinerio {} with r = {}".format(dataset_name, sinerio, real_r2))
        import PendulumGeneration_new
    else:
        print("3. Loading {} {} dataset with real_q = {} and real_r = {}".format(dataset_name,sinerio,real_q2, real_r2))
else:
    if data_gen_flag:
        print("3. Start data generation {} sinerio {} with q2 = {}".format(dataset_name, sinerio, sys_model.real_q2))
        [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataGen(sys_model,dataset_name,sinerio,T, T_test, N_E, N_CV, N_T,randomInit=False)  # taking time
    else:
        print("3. Loading {} {} dataset with real_q {} and real_r = {}".format(dataset_name, sinerio, real_q2, real_r2))

path_for_states = rf'./Simulations/{dataset_name}/states_q2_{real_q2}_{sinerio}.npz'
path_for_observations = rf'./Simulations/{dataset_name}/Observations_q2_{real_q2}_{sinerio}.npz'
###############################################################################################

################# Architecture knet ################################
if real_r2 == 0.5:
    prior_r2 = 4

if prior_flag:
    model_encoder_trained = Encoder_conv_with_prior(d)
    sinerio = sinerio+"_with_prior"
    path_enc = r'./Encoder/{}/{}/{}_Only_encoder_r={}_prior={}.pt'.format(dataset_name, sinerio, dataset_name, real_r2, prior_r2)
    model_encoder_trained.load_state_dict(torch.load(path_enc),strict=False)
    path_KNetLatent_trained = folder_KNetLatent_models + 'KNetLatent_optimal_' + dataset_name + '_' + sinerio + '_' + 'fix_enc_' + str(int(fix_encoder_flag)) + '_r_' + str(real_r2) + '.pt'
else:
    model_encoder_trained = Encoder_conv(d)
    if "Decimation" in sinerio:
      model_encoder_trained.load_state_dict(torch.load(folder_encoder_model + sinerio + '/' + dataset_name + '_Only_encoder_r={}.pt'.format(real_r2)),strict=False)
      path_KNetLatent_trained = folder_KNetLatent_models + 'KNatLatent_optimal_' + dataset_name + '_' + sinerio + '_fix_enc_' + str(int(fix_encoder_flag)) + '_r_' + str(real_r2) + '.pt'
    else:
      path_enc = r'./Encoder/{}/{}/{}_Only_encoder_r={}.pt'.format(dataset_name, sinerio, dataset_name, real_r2)
      model_encoder_trained.load_state_dict(torch.load(path_enc),strict=False)
      path_KNetLatent_trained = folder_KNetLatent_models + 'KNatLatent_optimal_' + dataset_name + '_' + 'Baseline_fix_enc_' + str(int(fix_encoder_flag)) + '_r_' + str(real_r2) + '.pt'