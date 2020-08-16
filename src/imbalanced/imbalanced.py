import numpy as np
import util
import sys
from sklearn.metrics import confusion_matrix
from random import random

sys.path.append('/Users/apple/OneDrive - Leland Stanford Junior University/Stanford/Course/2019 spring/CS 229/ps1/src/linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_test, y_test = util.load_dataset(validation_path, add_intercept = True)
    naive_model = LogisticRegression()
    # train the model
    naive_model.fit(x_train, y_train)
    # get the predictions prob and save it
    predictions = naive_model.predict(x_test)
    np.savetxt(output_path_naive, predictions)
    util.plot(x_test, y_test, naive_model.theta, output_path_naive[:-3] + "png")
    # if the prob is larger than 0.5 then tag y as 1
    y_pred = []
    for i in range(len(y_test)):
        if predictions[i] >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # calculate the a
    nm = confusion_matrix(y_test, y_pred)
    a1 = nm[1,1]/ (nm[1,0] + nm[1,1])
    a0 = nm[0,0] / (nm[0,1] + nm[0,0])
    a = (nm[1,1] +nm[0,0]) / (nm[1,0] + nm[1,0]+nm[0, 1] + nm[0, 0])
    a_ = (a1 + a0)/2
    print("a1=",a1)
    print("a0=", a0)
    print("a=", a)
    print("a_=", a_)
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    upsampling_model = LogisticRegression()
    new_x, new_y = newData(x_train, y_train, kappa)
    upsampling_model.fit(new_x, new_y)
    upPredictions = upsampling_model.predict(x_test)
    np.savetxt(output_path_upsampling, upPredictions)
    util.plot(x_test, y_test, upsampling_model.theta, output_path_upsampling[:-3] + "png")
    y_pred1 = []
    for i in range(len(y_test)):
        if upPredictions[i] >= 0.5:
            y_pred1.append(1)
        else:
            y_pred1.append(0)
    nm1 = confusion_matrix(y_test, y_pred1)
    a1 = nm1[1, 1] / (nm1[1, 0] + nm1[1, 1])
    a0 = nm1[0, 0] / (nm1[0, 1] + nm1[0, 0])
    a = (nm1[1, 1] + nm1[0, 0]) / (nm1[1, 0] + nm1[1, 0] + nm1[0, 1] + nm1[0, 0])
    a_ = (a1 + a0) / 2
    print("a1=", a1)
    print("a0=", a0)
    print("a=", a)
    print("a_=", a_)
    # *** END CODE HERE
# generate the new dataset
def newData(x_train, y_train, kappa):
    x_reweight = x_train.copy()
    y_reweight = y_train.copy()
    for i in range(len(y_train)):
        if y_train[i] == 1:
            for n in range(int(1 / kappa) - 1):
                y_reweight = np.append(y_reweight, y_train[i])
                x_reweight = np.append(x_reweight, np.expand_dims(x_train[i], axis=0), axis=0)
    return x_reweight, y_reweight


if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')

