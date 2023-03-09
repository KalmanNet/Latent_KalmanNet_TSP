import numpy as np
from PIL import Image
from PIL import ImageDraw
import os
import gc

class Pendulum:

    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    SIMULATION_LENGTH_KEY = 'simulation_length'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'
    rng = np.random

    def __init__(self,
                 img_size=28,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params=None,
                 seed=0):
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.random = np.random.RandomState(seed)

        # image parameters
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55
        self.plt_width = 8

        # simulation parameters
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()

        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]

        self.simulation_length = pendulum_params[Pendulum.SIMULATION_LENGTH_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std


    @staticmethod
    def pendulum_default_params():
        return {
            Pendulum.LENGTH_KEY: 0,
            Pendulum.GRAVITY_KEY: 9.81,
            Pendulum.SIMULATION_LENGTH_KEY: 0,
            Pendulum.DT_KEY: 0,
            Pendulum.SIM_DT_KEY: 0}

    def sample_continuous_data_set(self, num_episodes, seed=None):
        """

        :param num_episodes: number of episodes/trajectories created
        :param seed: for reproducibility
        :return: a multidimensional array dim: (num_episodes, 2 (position, velocity), number_steps (simulation_length/sim_dt))
        """
        self.Q = self.transition_noise_std ** 2 * np.array([[1, 0],[0, 1]])
                 #np.array([[1 / 3 * self.sim_dt, 1 / 2 * self.sim_dt ** 2],[1 / 2 * self.sim_dt ** 2, self.sim_dt]])
        #** 3
        print("Q is: {}".format(self.Q))
        print("sim_dt is: {}".format(self.sim_dt))
        print("length is: {}".format(self.length))
        episode_length = int(np.round(self.simulation_length/self.sim_dt))

        if seed is not None:
            self.random.seed(seed)
        states = np.zeros((num_episodes, self.state_dim, episode_length))
        states[:, :, 0] = self._sample_init_state(num_episodes)
        for i in range(1, episode_length):
            next_state = self._get_next_states(states[:, :, i - 1])
            states[:, :, i] = next_state

        return states

    def _sample_init_state(self, nr_epochs):
        """
        Randomly initialize the states of the pendulum (theta, omega), omega is always set to 0.
        :param nr_epochs: number of episodes/trajectories
        :return: return an array of initial values of shape (#episodes, 2)
        """
        #return np.concatenate((self.rng.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(nr_epochs, 1)),
        #                       np.zeros((nr_epochs, 1))), 1)
        return np.concatenate((np.ones((nr_epochs,1))* 90* np.pi / 180,
                               np.zeros((nr_epochs, 1))), 1)

    def _get_next_states(self, states):
        """
        Takes an array of states of dim: (num_episodes, 2) and using the discrete pendulum evolution equations computes
        the next states.
        :param states: array of previous states dim (num_episodes, 2)
        :return: array of next states dim (num_episodes, 2)
        """
        transition_noise = self.rng.multivariate_normal(np.array([0, 0]), self.Q, size=states.shape[0])
        pos_new = states[:, 0:1] + self.sim_dt * states[:, 1:2] - 0.5 * self.g/self.length * self.sim_dt**2 * np.sin(states[:, 0:1])+ transition_noise[:, :1]
        vel_new = states[:, 1:2] - self.g/self.length * self.sim_dt * np.sin(states[:, 0:1])+ transition_noise[:, 1:]

        states = np.concatenate((pos_new, vel_new), axis=1)
        return states

    def decimate_data(self, states):
        """
        Takes as input the fine grained trajectory generated with sample_continuous_data_set, and subsamples it using
        taking one sample every dt/sim_dt samples.
        :param states: array of fine grained trajectories
        :return: array of coarse graind trajectory (decimated)
        """
        step = int(np.round(self.dt/self.sim_dt))
        return states[:, :, ::step]

    # def add_observation_noise(self, states):
    #     """
    #     Adds gaussian noise on top of the observation
    #     :param states: array of trajectories
    #     :return: array of trajectories with observation noisy on top
    #     """
    #     states += self.observation_noise_std * self.rng.standard_normal(states.shape)
    #     return states

    def generate_images(self, states):
        cartesian = self.pendulum_kinematic(states)
        images = self._generate_images(cartesian)
        return images

    def pendulum_kinematic(self, js_batch):
        theta = js_batch[:, :1, :]
        x = np.sin(theta) * self.length
        y = np.cos(theta) * self.length
        return np.concatenate([x, y], axis=1)

    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos[:, 0, :].shape) + [self.img_size, self.img_size], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]): #running over trajectories
            print(seq_idx)
            for idx in range(ts_pos.shape[2]): #running over time steps
                imgs[seq_idx, idx, ...] = self._generate_single_image(ts_pos[seq_idx, :,  idx])
            # import matplotlib.pyplot as plt
            # rows, cols = 2, 2
            # plt.subplot(rows, cols, 1)
            # plt.imshow(imgs[seq_idx, 0, ...], cmap='gray')
            # plt.title("x={}, y={}".format(np.round(ts_pos[seq_idx,0,0],2), np.round(ts_pos[seq_idx,1,0],2)))
            # plt.subplot(rows, cols, 2)
            # plt.imshow(imgs[seq_idx, 4, ...], cmap='gray')
            # plt.title("x={}, y={}".format(np.round(ts_pos[seq_idx, 0, 4],2), np.round(ts_pos[seq_idx, 1, 4],2)))
            # plt.subplot(rows, cols, 3)
            # plt.imshow(imgs[seq_idx, 10, ...], cmap='gray')
            # plt.title("x={}, y={}".format(np.round(ts_pos[seq_idx, 0, 10],2), np.round(ts_pos[seq_idx, 1, 10],2)))
            # plt.subplot(rows, cols, 4)
            # plt.imshow(imgs[seq_idx, 15, ...], cmap='gray')
            # plt.title("x={}, y={}".format(np.round(ts_pos[seq_idx, 0, 15],2), np.round(ts_pos[seq_idx, 1, 15],2)))
            # plt.show()
        return imgs

    def _generate_single_image(self, pos):
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        image = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(image)
        draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
        image = image.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(image)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array

if __name__ == "__main__":

    img_size = 28
    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.SIM_DT_KEY] = 5e-3
    pend_params[Pendulum.LENGTH_KEY] = 1
    pend_params[Pendulum.DT_KEY] = 5e-2
    pend_params[Pendulum.SIMULATION_LENGTH_KEY] = 2 #number of "side to side"
    data = Pendulum(img_size=img_size,
                    pendulum_params=pend_params,
                    seed=0)
    training_set_size = 1000
    validation_set_size = 100
    test_set_size = 100

    q2 = 0.001
    r2 = 0.3
    print(f"Generating data for q2: {q2}")
    data.transition_noise_std = np.sqrt(q2)
    continuous = data.sample_continuous_data_set(training_set_size + validation_set_size + test_set_size)
    np.random.shuffle(continuous)
    os.makedirs(r"Simulations/Pendulum/", exist_ok=True)
    np.savez(rf"Simulations/Pendulum/states_q2_{q2}_r2_{r2}.npz",
             training_set=continuous[:training_set_size, :, :],
             validation_set=continuous[training_set_size:(training_set_size + validation_set_size), :, :],
             test_set=continuous[training_set_size + validation_set_size:, :, :])
    img = data.generate_images(continuous)
    img = img/255
    img += r2*np.random.standard_normal(img.shape)
    np.savez(rf"Simulations/Pendulum/observations_q2_{q2}_r2_{r2}.npz",
             training_set=img[:training_set_size, ...],
             validation_set=img[training_set_size:(training_set_size + validation_set_size), ...],
             test_set=img[(training_set_size + validation_set_size):, ...])

    import matplotlib.pyplot as plt
    target = continuous[6,0,:]
    fig, ax = plt.subplots()
    ax.plot(target, marker='v', label='Target', color='black')

    plt.title(r'Pendulum Estimated trajectory')
    plt.xlabel(r'Time steps')
    plt.ylabel('Angle')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    #del img
    #gc.collect()

    #data.observation_noise_std = r
    #noisy = img + data.observation_noise_std * data.rng.standard_normal(img.shape)
    #os.makedirs(r"Datasets\Pendulum_same_init_45\high_resolution_noisy_data/", exist_ok=True)
    #np.savez(rf"Simulations/Pendulum/Observations_r2_{r2}.npz",
    #         training_set=noisy[:training_set_size, :, :],
    #         validation_set=noisy[training_set_size:(training_set_size + validation_set_size), :, :],
    #         test_set=noisy[(training_set_size + validation_set_size):, :, :],
    #         q2=q2, sim_dt=data.sim_dt, dt=data.dt, simulation_length=data.simulation_length, r2=r2)
    itay=29





