import torch
import numpy as np
g = 9.81
length = 1.0
sim_dt = 0.05
m = 2
#H = torch.eye(m)
H = torch.from_numpy(np.array([1,0])).unsqueeze(0).float()
y_size = 28
n = y_size*y_size
m1x_0 = torch.from_numpy(np.array([90* torch.pi / 180,0])).unsqueeze(1)
m2x_0 = 0 * 0 * torch.eye(m)
T = 20
T_test = 400
d=1
real_q2 = 0.001
prior_r2=8

def f_function(prev_state):
    pos_new = prev_state[0] + sim_dt * prev_state[1] - 0.5 * g / length * sim_dt ** 2 * torch.sin(prev_state[0])
    vel_new = prev_state[1] - g/ length * sim_dt * torch.sin(prev_state[0])
    next_state = torch.stack((pos_new, vel_new), axis=0).transpose(0,1).double()
    return next_state

# not used
def h_function(state):
    return 1