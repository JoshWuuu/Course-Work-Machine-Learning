import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(X.T @ X, y @ X)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        phi_x = np.zeros((X.shape[0], k+1))
        for i in range(X.shape[0]):
            for j in range(k + 1):
                phi_x[i][j] = X[i][1] ** j   # phi(x) = x^0 + x^1 + ... + x^k
        return phi_x
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        phi_x = np.zeros((X.shape[0], k + 2))
        for i in range(X.shape[0]):
            for j in range(k + 1):
                phi_x[i][j] = X[i][1] ** j
            phi_x[i][k + 1] = np.sin(X[i][1])  # phi(x) = x^0 + x^1 + ... + x^k + sin(x)
        return phi_x
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # phi_x = self.create_poly(self, 3, X)
        # return self.theta.dot(phi_x)
        return X @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    # plot training data
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    # plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    # train
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        linear_reg = LinearModel()

        # Transfer the input data sets into k-dimensional vectors (K-degree polynomial)
        if sine:
            train_phi = linear_reg.create_sin(k, train_x)
            plot_phi = linear_reg.create_sin(k, plot_x)
        else:
            train_phi = linear_reg.create_poly(k, train_x)
            plot_phi = linear_reg.create_poly(k, plot_x)

        # Train the Linear Regression Model
        linear_reg.fit(train_phi, train_y)
        plot_y = linear_reg.predict(plot_phi)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Question 4(b)
    run_exp(train_path, False, [3], filename='4_B.png')
    # Question 4(c)
    run_exp(train_path, False, [3, 5, 10, 20], filename='4_C.png')
    # Question 4(d)
    run_exp(train_path, True, [0, 1, 2, 3, 5, 10, 20], filename='4_D.png')
    # Question 4(e)
    run_exp(small_path, False, [1, 2, 5, 10, 20], filename='4_E.png')
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
