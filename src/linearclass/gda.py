import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    model = GDA()
    # train the GDA model
    model.fit(x_train, y_train)
    # load test data
    x_test, y_test = util.load_dataset(valid_path, add_intercept=False)
    # predict the test data by trained model
    test_prob = model.predict(x_test)
    theta = np.concatenate((model.theta_0, model.theta_1.T), axis=0)
    util.plot(x_test, y_test, theta, save_path[:-3] + "png")
    np.savetxt(save_path, test_prob)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # n is sample size, d is the feature count
        n ,d = x.shape
        self.phi = np.mean(y)
        self.mu_0 = np.mean(x[y==0], axis=0).reshape(d, 1) # 2 features y =0
        self.mu_1 = np.mean(x[y==1], axis=0).reshape(d, 1) # y = 1
        # 2 by 2 matrix row is x1 x2 col 0 1
        total = np.zeros([d, d])
        for i in range(n):
            # y lable is equal to 1
            if (y[i] == 1):
                difference = x[i, :].reshape(d, 1) - self.mu_1
                total += np.matmul(difference, difference.T)
            # y lable is equal to 0
            elif (y[i] == 0):
                difference = x[i, :].reshape(d, 1) - self.mu_0
                total += np.matmul(difference, difference.T)

        self.sigma = total/n
        # theta with the x 2 BY 1 matrix
        self.theta_1 = np.matmul((self.mu_1-self.mu_0).T, np.linalg.inv(self.sigma))
        # constant term theta 1 by 1 matrix
        self.theta_0 = (0.5 * ((np.matmul(np.matmul(self.mu_0.T, np.linalg.inv(self.sigma)), self.mu_0))
                        - (np.matmul(np.matmul(self.mu_1.T, np.linalg.inv(self.sigma)), self.mu_1)))
                        + np.log(self.phi / (1 - self.phi)))


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        self.func_y1 = 1 / (1 + np.exp(-(np.matmul(x, self.theta_1.T) + np.matmul(np.ones([n, 1]), self.theta_0))))
        self.prob = self.func_y1.reshape(n,)
        return self.prob
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
