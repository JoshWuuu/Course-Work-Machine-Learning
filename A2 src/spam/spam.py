from collections import OrderedDict

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # split the message and return a list
    words = message.split(" ")
    return [word.lower() for word in words]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_dic = {}
    result_dic = {}
    for message in messages:
        # set() [2,2,4] to [2,4], prevent word to double count
        words = set(get_words(message))
        # add the occurence of each word into dict
        for word in words:
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
    i = 0
    # if word # is larger than 5, add to result
    for word in word_dic:
        if word_dic[word] >= 5:
            result_dic[word] = i
            i += 1
    return result_dic
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # create an zero array
    result_array = np.zeros([len(messages), len(word_dictionary)])
    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            # if word in word_dict, then plus 1
            if word in word_dictionary:
                j = word_dictionary[word]
                result_array[i, j] += 1
    return result_array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # Utilize a 2 by p+1(dictionary size + 1 py=1 and py = 0) matrix as the fitted model
    n, p = matrix.shape
    result = np.zeros((2, p+1))
    # y_0 y_1 is the total number of words in y == 0 and y == 1 respectively
    y_0 = np.sum(np.sum(matrix[labels == 0], axis=0))
    y_1 = np.sum(np.sum(matrix[labels == 1], axis=0))
    p_y0 = len(matrix[labels == 0]) / n
    p_y1 = len(matrix[labels == 1]) / n
    result[0,0] = p_y0
    result[1,0] = p_y1
    # loop thru each column
    for i in range(p):
        x = matrix[:,i]
        sumX_y0 = np.sum(x[labels==0])
        sumX_y1 = np.sum(x[labels==1])
        # apply laplace
        prob_y0 = (sumX_y0+1)/(p+y_0)
        prob_y1 = (sumX_y1+1)/(p+y_1)
        result[0,i+1] = prob_y0
        result[1,i+1] = prob_y1
    return result
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    n, p = matrix.shape
    result = np.zeros(n)
    # loop thru each mail
    for i in range(n):
        prob_y0x = 1
        prob_y1x = 1
        for j in range(p):
            if (matrix[i, j] != 0):
                prob_y0x *= pow(model[0, j+1], matrix[i, j])
                prob_y1x *= pow(model[1, j+1], matrix[i, j])
        prob_y0 = prob_y0x * model[0,0] / (prob_y0x * model[0,0] + prob_y1x * model[1,0])
        prob_y1 = prob_y1x * model[1,0] / (prob_y0x * model[0,0] + prob_y1x * model[1,0])
        if prob_y0 > prob_y1:
            result[i] = 0
        else:
            result[i] = 1
    return result
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    result = {}
    for word, index in dictionary.items():
        prob = np.log(model[1,index+1] / model[0,index+1])
        result[word] = prob
    #  sort the dict by the value
    result1 = {k: v for k, v in sorted(result.items(), key=lambda item: -item[1])}
    top_five = []
    for key in result1:
        top_five.append(key)
        if len(top_five) == 5:
            break
    return top_five
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    result = {}
    for i in radius_to_consider:
        svm_model = svm.svm_train(train_matrix, train_labels, i)
        svm_predict = svm.svm_predict(svm_model, val_matrix, i)
        accuracy = np.mean(svm_predict == val_labels)
        result[i] = accuracy
    result1 = {k: v for k, v in sorted(result.items(), key=lambda item: -item[1])}
    return list(result1.keys())[0]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix.txt', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    np.savetxt('fit_result.txt', naive_bayes_model)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions.txt', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
