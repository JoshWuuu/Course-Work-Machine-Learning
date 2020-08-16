import numpy as np
import matplotlib.pyplot as plt
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    logistic_reg = LogisticRegression()
    logistic_reg.fit(x_train, y_train)

    # Import the validation data
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # Plot the validation data and the decision boundary
    util.plot(x_valid, y_valid, logistic_reg.theta, save_path[:-3] + "png")
    plt.show()

    # Use np.savetxt to save predictions on eval set to save_path
    predictions = logistic_reg.predict(x_train)
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        dtheta = 1
        iter = 0
        self.theta = np.zeros(x[0].shape)
        while dtheta > self.eps and iter < self.max_iter:
            # the logistic function g(z)
            z = x.dot(self.theta)
            g_z = 1 / (1 + np.exp(-z))

            # the vector of partial derivatives of l(theta)
            grad = np.mean((y - g_z) * x.T, axis=1)

            # the Hessian
            Hessian = np.zeros((x.shape[1], x.shape[1]))
            for i in range(Hessian.shape[0]):
                for j in range(Hessian.shape[0]):
                    Hessian[i][j] = -np.mean(g_z * (1 - g_z) * x[:, i] * x[:, j])

            theta_prev = self.theta.copy()                      # theta(k-1)
            self.theta -= np.linalg.inv(Hessian).dot(grad)      # theta(k)
            dtheta = np.linalg.norm(self.theta - theta_prev)    # ||theta(k) - theta(k-1)||
            iter += 1
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
