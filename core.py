import numpy as np

def initialize_weights(network_dimensions):
    """Initialize the weights in the network.

    Initialize the weights in the network using a gaussian distribution of mean
    0.0 and of standard deviation 0.5 for each of its weights. Information on
    the layout of the network is provided in network_dimensions.

    Parameters
    ----------
    network_dimensions : sequence of int
        Sequence of dimensions describing the size of each layer within the
        network, from the input layer to the output layer. Beware, any layer
        using a bias value must have it included in the corresponding dimension
        value comprised in network_dimensions.

    Returns
    -------
    A list of len(network_dimensions) numpy.matrix objects where each
    numpy.matrix represents the weights that connect one layer to the next one.
    When a layer is the input of another layer in the feed-forward network,
    its size matches the number of columns of the corresponding
    numpy.matrix in the list, whereas the number of rows of this very matrix
    corresponds to the size of the layer which it feeds.
    """
    variance = 0.5**2
    network = [
        np.asmatrix(np.reshape(
            np.random.multivariate_normal(np.zeros(n*p), variance*np.eye(n*p)),
            (p,n)))
        for (n,p) in zip(network_dimensions[:-1], network_dimensions[1:])
    ]
    return network

def perceptron_learning(initial_network, input_values, target_values, step_size,
                        batch_learning=True, max_epochs=100):
    """Apply the perceptron learning algorithm to a given network.

    Apply the perceptron learning algorithm to the network described and
    initialised by initial_network. The training set is composed of input_values
    and target_values. step_size will be used as step size to control the
    convergence speed. The algorithm can be performed in batch learning mode,
    in which case all of the misclassifications performed on the training set
    after each epoch will be used in one shot to update the weights. If the
    batch mode is disabled, the algorithm iterates over all input values and
    their associated target values to update the weights (when given a new input
    pattern, it does not update the weights repeatedly until this pattern is
    correctly classified before moving onto another one). If the data is not
    linearly separable or if convergence is too slow, max_epochs can be used
    to abort learning.

    Parameters
    ----------
    initial_network : list of numpy.matrix
        A list of 1 numpy.matrix representing the weights connecting the input
        layer to the output layer. The weights used to control the bias must
        be included in the last column of this matrix.
    input_values : numpy.ndarray
        Input patterns without the extra row used for the biases in the network.
    target_values : numpy.ndarray
        Target values corresponding to input_values and whose components equal
        -1 or 1.
    step_size : float
    batch_learning : bool
    max_epochs : int

    Returns
    -------
    list of numpy.matrix, list of int
        The weights obtained after learning at each epoch, contained in a list
        of numpy.matrix where each numpy.matrix corresponds to one epoch;
        and a list of classification errors at each epoch of learning.
    """
    input_values = np.append(input_values, np.ones((1, input_values.shape[1])),
                             axis=0)
    input_values = np.asmatrix(input_values)
    target_values = np.matrix(target_values, copy=True)
    target_values[target_values == -1] = 0

    epoch = 0
    weights = np.matrix(initial_network[0], copy=True)
    output_values = step_function(weights * input_values)
    weights_history = [np.matrix(weights, copy=True)]
    misclassification_history = [count_misclassified_patterns(output_values,
                                                              target_values)]
    while (epoch <= max_epochs):
        if batch_learning:
            weights += (step_size
                        * (target_values - step_function(output_values))
                        * input_values.T)
        else:
            for i in range(input_values.shape[1]):
                x = input_values[:, i]
                t = target_values[:, i]
                weights += step_size * (t - step_function(weights*x)) * x.T

        epoch += 1
        output_values = step_function(weights * input_values)
        misclassification_history += [
            count_misclassified_patterns(output_values, target_values)]
        weights_history += [np.matrix(weights, copy=True)]
        if misclassification_history[-1] == 0 : break

    return weights_history, misclassification_history

def step_function(weighted_sums):
    """Apply the step function to the given set of output vectors.

    Parameters
    ----------
    weighted_sums : numpy.matrix
        Set of output vectors (one vector per column). Each component of each
        vector is set to 1 if greater or equal to 0, or is set to 0 otherwise.
        weighted_sums is modified as a side-effect of this function.

    Returns
    -------
    weighted_sums afer the changes have been applied.
    """
    weighted_sums[weighted_sums >= 0.] = 1
    weighted_sums[weighted_sums < 0.] = 0
    return weighted_sums

def count_misclassified_patterns(output_values, target_values):
    """Return the absolute number of misclassified samples.

    Parameters
    ----------
    output_values : numpy.matrix
        Values from output layer of the network. There must be one output vector
        per column.
    target_values : numpy.matrix
        Target vectors from the training data. There must be one vector per
        column.

    Returns
    -------
    int
        Number of columns in output_values which do not match the corresponding
        column in target_values.
    """
    differences = output_values - target_values
    errors = np.linalg.norm(differences, axis=0) != 0
    return np.sum(errors)
