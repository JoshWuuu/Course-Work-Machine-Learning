import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    mu = []
    sigma = []
    n = x.shape[0]
    random = np.random.randint(0,K,n)
    for i in range(K):
        i_Cluster = x[random == i]
        mu.append(np.mean(i_Cluster, axis=0))
        sigma.append(np.cov(i_Cluster, rowvar=False))
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones([K]) / K
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones([n, K]) / K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        if it % 5 == 0:
            print("number of iterations" + str(it))
        prev_ll = ll
        ll = 0
        it += 1
        # (1) E-step: Update your estimates in w
        n, k = w.shape
        # loop thru each point and find the corresponding j cluster prob
        for i in range(n):
            denomi = 0
            for j in range(k):
                numera = multivariate_normal.pdf(x[i, :], mu[j], sigma[j]) * phi[j]
                denomi += numera
                w[i,j] = numera
            w[i,] /= denomi
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # phi
        sumW = np.sum(w, axis=0)
        phi = 1 / n * sumW
        # calculate each j mu and sigma first and add them into mu sigma
        n ,d = x.shape
        for j in range(k):
            sigma_j = np.zeros((d, d))
            mu_j = np.zeros(d)
            for i in range(n):
                mu_j += (x[i, :] * w[i, j])
                dif = (x[i, :] - mu[j]).reshape(-1, 1)
                sigma_j += w[i, j] * (dif) @ (dif).T
            mu[j] = mu_j
            sigma[j] = sigma_j
        mu = mu / sumW.reshape((4,1))
        sigma = sigma / sumW.reshape((4,1,1))
        # (3) Compute the log-likelihood of the data to check for convergence.
        for i in range(n):
            log_xi = 0
            for j in range(k):
                log_xi += np.log(multivariate_normal.pdf(x[i,:], mu[j], sigma[j]) * phi[j])
            ll += log_xi
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        if it % 5 == 0:
            print("number of iterations" + str(it))
        prev_ll = ll
        ll = 0
        it += 1
        # (1) E-step: Update your estimates in w
        n, k = w.shape
        # loop thru each point and find the corresponding j cluster prob
        for i in range(n):
            denomi = 0
            for j in range(k):
                numera = multivariate_normal.pdf(x[i, :], mu[j], sigma[j]) * phi[j]
                denomi += numera
                w[i, j] = numera
            w[i,] /= denomi
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # count the number of each label cluster
        labelCluster = np.zeros(k)
        for j in range(k):
            v = x_tilde[z_tilde[:,0] == j]
            a, p = v.shape
            labelCluster[j] = a
        # phi
        m ,l = x_tilde.shape
        sumW = np.sum(w, axis=0)
        phi = (sumW + alpha * labelCluster) / (n + alpha * m)
        # calculate each j mu and sigma first and add them into mu sigma
        n ,d = x.shape
        for j in range(k):
            sigma_j = np.zeros((d, d))
            mu_j = np.zeros(d)
            for i in range(n):
                mu_j += (x[i, :] * w[i, j])
                dif = (x[i, :] - mu[j]).reshape(2, 1)
                sigma_j += w[i, j] * (dif) @ (dif).T
            dif_lab = x_tilde[z_tilde[:, 0] == j] - mu[j].reshape(1, 2)
            sigma[j] = sigma_j + alpha * (dif_lab).T @ (dif_lab)
            mu[j] = mu_j + alpha * np.sum(x_tilde[z_tilde[:,0] == j], axis=0)
        mu = mu / (sumW.reshape((4, 1)) + alpha * labelCluster.reshape((4, 1)))
        sigma = sigma / (sumW.reshape((4, 1, 1)) + alpha * labelCluster.reshape((4, 1, 1)))
        # (3) Compute the log-likelihood of the data to check for convergence.
        ll_unsup = 0
        for i in range(n):
            log_xi = 0
            for j in range(k):
                log_xi += np.log(multivariate_normal.pdf(x[i, :], mu[j], sigma[j]) * phi[j])
            ll_unsup += log_xi
        ll_sup = 0
        for i in range(m):
            sup_z = int(z_tilde[i])
            ll_sup += np.log(multivariate_normal.pdf(x_tilde[i, :], mu[sup_z], sigma[sup_z]) * phi[sup_z])
        ll = ll_unsup + alpha * ll_sup
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
