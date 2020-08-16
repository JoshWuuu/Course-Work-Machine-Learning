import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    model = PoissonRegression()
    # train the data
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    # test the validation set
    pred = model.predict(x_test)
    np.savetxt(save_path, pred)
    # plot it
    plt.scatter(x = y_test, y = pred)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.show()
    plt.savefig(save_path[:-3] + "png")
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        theta_norm = 1
        iter = 0
        # initialize the theta as zero d by 1 matrix
        self.theta = np.zeros(d)
        # if norm is still larger criteria and the iter count is still smaller that criteria
        while theta_norm > self.eps and iter < self.max_iter:
            # sum all the difference
            total = np.zeros(d)
            for i in range(n):
                total += (y[i] - np.exp(np.matmul(self.theta, x[i, :]).T)) * x[i, :]
            # gradient assend equation
            theta1 = self.theta + self.step_size * total
            # norm
            theta_norm = np.linalg.norm(theta1 - self.theta)
            self.theta = theta1
            iter += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(np.matmul(self.theta, x.T))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
