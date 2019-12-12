# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/ directory of this source tree.
#
# This code has been modified from the original version at
# https://github.com/inspire-group/RON-Gauss/blob/master/ron_gauss.py
# Modifications copyright (C) 2019-present, Royal Bank of Canada.

# ron_gauss.py implements the RON_GAUSS generative model to generate private synthetic data
import numpy as np
import scipy
from sklearn import preprocessing


class RONGauss:

    def __init__(self, z_dim, target_epsilon, target_delta, conditional):
        self.epsilon_mean = target_epsilon / 2
        self.epsilon_cov = target_epsilon / 2
        self.delta_mean = target_delta / 2
        self.delta_cov = target_delta / 2
        self.z_dim = z_dim
        self.conditional = conditional

    def generate(
            self,
            X,
            y=None,
            n_samples=None,
            reconstruct=True,
            centering=True,
            prng_seed=None,
            max_y = None,
    ):

        (n, m) = X.shape
        if n_samples is None:
            n_samples = n

        if self.conditional:
            return self._gmm_rongauss(X, y, n_samples, reconstruct, prng_seed)

        else:
            return self._supervised_rongauss(X, y, n_samples, max_y, reconstruct, centering, prng_seed)


    def _gmm_rongauss(
            self,
            X,
            y,
            n_samples,
            reconstruct,
            prng_seed,
    ):
        prng = np.random.RandomState(prng_seed)
        syn_x = None
        syn_y = np.array([])
        dp_mean_dict = {}
        for label in np.unique(y):
            idx = np.where(y == label)
            x_class = X[idx]
            (x_bar, mu_dp) = self._data_preprocessing(x_class, self.epsilon_mean, prng)
            dp_mean_dict[label] = mu_dp
            (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, self.z_dim, prng)

            (n, p) = x_tilda.shape
            mu_dp_tilda = np.inner(mu_dp, proj_matrix)
            cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n

            # Add gaussian noise
            c = np.sqrt(2 * np.log(1.25 / self.delta_cov))
            b = (c * 2.) / (n * self.epsilon_cov)
            noise = np.random.normal(scale=b, size=(p, p))

            cov_dp = cov_matrix + noise
            synth_data = prng.multivariate_normal(mu_dp_tilda, cov_dp, n_samples)

            if reconstruct:
                synth_data = self._reconstruction(synth_data, proj_matrix)
            if syn_x is None:
                syn_x = synth_data
            else:
                syn_x = np.vstack((syn_x, synth_data))

            syn_y = np.append(syn_y, label * np.ones(n_samples))

        return syn_x, syn_y, dp_mean_dict

    def _supervised_rongauss(
            self,
            X,
            y,
            n_samples,
            max_y,
            reconstruct,
            centering,
            prng_seed,
    ):

        prng = np.random.RandomState(prng_seed)
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean, prng)
        (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, self.z_dim, prng)

        (n, p) = x_tilda.shape

        y_reshaped = y.reshape(len(y), 1)
        augmented_mat = np.hstack((x_tilda, y_reshaped))
        cov_matrix = np.inner(augmented_mat.T, augmented_mat.T) / n

        # Add gaussian noise
        c = np.sqrt(2 * np.log(1.25 / self.delta_cov))
        b = (c * (2.0 + 4.0 * max_y + max_y ** 2)) / (n * self.epsilon_cov)
        noise = np.random.normal(scale=b, size=(p + 1, p + 1))

        cov_dp = cov_matrix + noise

        synth_data = prng.multivariate_normal(np.zeros(p + 1), cov_dp, n_samples)
        x_dp = synth_data[:, 0:-1]
        y_dp = synth_data[:, -1]
        if reconstruct:
            x_dp = self._reconstruction(x_dp, proj_matrix)
        else:
            # project the mean down to the lower dimention
            mu_dp = np.inner(mu_dp, proj_matrix)
        self._mu_dp = mu_dp

        if not centering:
            x_dp = x_dp + mu_dp

        return x_dp, y_dp, mu_dp




    @staticmethod
    def _data_preprocessing(X, epsilon_mean, prng=None):
        if prng is None:
            prng = np.random.RandomState()
        (n, m) = X.shape
        # pre-normalize
        x_norm = preprocessing.normalize(X)
        # derive dp-mean
        mu = np.mean(x_norm, axis=0)
        noise_var_mu = np.sqrt(m) / (n * epsilon_mean)
        laplace_noise = prng.laplace(scale=noise_var_mu, size=m)
        dp_mean = mu + laplace_noise
        # centering
        x_bar = x_norm - dp_mean
        # re-normalize
        x_bar = preprocessing.normalize(x_bar)
        return x_bar, dp_mean

    def _apply_ron_projection(self, x_bar, dimension, prng=None):
        (n, m) = x_bar.shape
        full_projection_matrix = self._generate_ron_matrix(m, prng)
        ron_matrix = full_projection_matrix[0:dimension]  # take the rows
        x_tilda = np.inner(x_bar, ron_matrix)
        return x_tilda, ron_matrix

    def _reconstruction(self, x_projected, ron_matrix):
        x_reconstructed = np.inner(x_projected, ron_matrix.T)
        return x_reconstructed

    def _generate_ron_matrix(self, m, prng=None):
        if prng is None:
            prng = np.random.RandomState()
        # generate random matrix
        random_matrix = prng.uniform(size=(m, m))
        # QR factorization
        q_matrix, r_matrix = scipy.linalg.qr(random_matrix)
        ron_matrix = q_matrix
        return ron_matrix
