## model Lorenz ##
import math
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import numpy as np
import torch.nn as nn
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    #print("Running on the CPU")

################for encoded transition ###################
class h_latent_space(nn.Module):
    def __init__(self, weights, bias, d,m):
        super(h_latent_space, self).__init__()
        self.fc = nn.Linear(m, d)
        with torch.no_grad():
            self.fc.weight.copy_(weights)
            self.fc.bias.copy_(bias.reshape(bias.shape[0]))
        # self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x1 = self.fc(x.double())
        # x2 = self.fc2(x1)
        return x1

############# for EKF #####################
#### dinamic process F matrix
y_size = 28
m =3
m1x_0 = torch.ones(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)

#### obzevation process H matrix
H = torch.eye(m)
b = torch.tensor([[0.0],
                  [0.0],
                 [0.0]])
n = y_size*y_size

B = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)]).float()
C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()
d=3
real_q2=0.1

# Baseline
J=5
prior_r2=1
#or prior_r2=4 for r2=0.5
delta_t = 0.02

# missmatch F
#J = 1
#prior_r2=1
# Decimatiom
#prior_r2 = 3

def f_function(x_prev): # return F matrix depand on x_prev
    A = (torch.add(torch.reshape(torch.matmul(B, x_prev.float()), (m, m)).T, C)).to(dev)
    # Taylor Expansion for F
    F = torch.eye(m)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)
    return F

# def create_trajectory_images(trajectory):
#     images = []
#     for point in trajectory:
#         x, y, z = point
#         # simulate image capture at (x, y, z)
#         image = capture_image(x, y, z)
#         images.append(image)
#     return images

def h_function(state):
    x = np.round(state[0].item(), 3)
    y = np.round(state[1].item(), 3)
    z = np.round(state[2].item(), 3)
    #x_min, x_max = [-30,30]
    #y_min, y_max = [-30,30]
    #z_min, z_max = [-10,60]

    # Normalize the x, y positions
    #x = (x - x_min) / (x_max - x_min)
    #y = (y - y_min) / (y_max - y_min)

    # Rescale the variance
    #z = (z - z_min) / (z_max - z_min)
    z = z/2 + 7
    # shift the distribution center to the center of the 28x28 image
    #x = x * 28
    #y = y * 28

    X = np.linspace(-30, 30, 28)
    Y = np.linspace(-40, 40, 28)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([x, y])
    Sigma = np.array([[z, 0], [0, z]])
    #+np.eye(2)*1e-3

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    Z = np.round(Z,3)
    Z = (Z-Z.min())/(Z.max()-Z.min())
    if np.isnan(Z).any():
        Itay=29
    return Z


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N



# def h_function(x):
#     x1 = np.round(x[0].item(),3)
#     x2 = np.round(x[1].item(),3)
#     x3 = np.round(x[2].item(),3)
#     y_sample = torch.zeros(y_size, y_size).double()
#     for i in range(y_size):
#         for j in range(y_size):
#             mone = np.round(np.math.pow((i-x1),2)+math.pow((j-x2),2),3)
#             mechane = 2*x3
#             y_sample[i, j] =np.round(np.exp(-mone/mechane),3)
#     return y_sample

# def get_h_derivative(x):
#     x1 = np.round(x[0].item(),3)
#     x2 = np.round(x[1].item(),3)
#     x3 = np.round(x[2].item(),3)
#     #sigma=0.2
#     H_drivative = torch.zeros(y_size, y_size, m).double()
#     for i in range(y_size):
#         for j in range(y_size):
#             mone = np.round(math.pow((i-x1),2)+math.pow((j-x2),2),3)
#             mechane = 2*x3
#             exp = np.round(10*np.exp(-mone/mechane),3)
#             H_drivative[i, j, 0]= np.round(exp*(i-x1)/x3,3)
#             H_drivative[i, j, 1] = np.round(exp*(j-x2)/x3,3)
#             H_drivative[i, j, 2] = np.round(exp*mone/(2*math.pow(x3,2)),3)
#     return H_drivative
