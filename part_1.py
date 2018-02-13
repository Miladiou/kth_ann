import os

import matplotlib.pyplot as plt
import numpy as np

import core

def generate_2_class_data(means, covariances, class_size):
    """Generate two classes of gaussian data.

    Parameters
    ----------
    means : list of 1-D array-like objects
        List of means for each class. Must be of size 2.
    covariances : list of 2-D array-like objects
        List of covariance matrices for each class. Must be of size 2.
    class_size : int
        Number of patterns in each class. Must be greater or equal to 1.

    Returns
    -------
    list of numpy.ndarray
        A list of numpy.ndarray of size 2. Each numpy.ndarray represents a
        class and
        contains one row per input pattern, the first columns containing the
        input value and the last column containing the associated target value
        (either -1 or 1).
    """
    data = [np.random.multivariate_normal(means[0], covariances[0], class_size)]
    data[0] = np.append(data[0], np.full((class_size, 1), -1.), axis=1)
    data += [np.random.multivariate_normal(means[1], covariances[1],
                                           class_size)]
    data[1] = np.append(data[1], np.ones((class_size, 1)), axis=1)
    return data

def shuffle_and_save_2_class_data(data, filename):
    """Shuffle the data and export it to a local file.

    Concatenate the two ndarrays in data, shuffle the rows of the resulting
    bigger ndarray and save its contents to the disk using filename.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of numpy.ndarray of size 2. Please see generate_2_class_data() for
        more information about its format.
    filename : string
        Name of the file in which the data will be saved.
    """
    data = np.append(data[0], data[1], 0)
    np.random.shuffle(data)
    np.save(filename, data)

def extract_2_class_data(filename):
    """Return the data saved with shuffle_and_save_2_class_data()."""
    return np.load(filename)

def plot_2_class_data(data, separating_hyperplane=None):
    """Plot the data extracted by extract_2_class_data().

    Plot data using red 'x' markers for the negative class and blue '+' markers
    for the positive class. If a separating hyperplane is given, it is plotted
    in yellow.

    Parameters
    ----------
    data : numpy.ndarray
        Data to plot with input values on the two first columns and target
        values on the last column.
    separating_hyperplane : numpy.matrix
        Weights representing a separating hyperplane. The numpy.matrix must have
        (1, 3) as shape.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure = plt.figure()
    axes = figure.add_subplot(111)

    negative_indices = data[:, 2] == -1
    positive_indices = data[:, 2] == 1
    axes.plot(data[negative_indices, 0], data[negative_indices, 1],
              linestyle="None", marker="x", markeredgecolor="r",
              markerfacecolor="r")
    axes.plot(data[positive_indices, 0], data[positive_indices, 1],
              linestyle="None", marker="+", markeredgecolor="b",
              markerfacecolor="b")

    if separating_hyperplane is not None:
        line_abscissas = np.arange(data.min(0)[0], data.max(0)[0], 0.1)
        slope = -separating_hyperplane[0, 0] / separating_hyperplane[0, 1]
        bias = -separating_hyperplane[0, 2] / separating_hyperplane[0, 1]
        axes.plot(line_abscissas, slope*line_abscissas + bias, "y-")

    axes.set_xlim(-3., 3.)
    axes.set_ylim(-3., 3.)
    axes.set_aspect("equal")

    return figure

def plot_classification_error(classification_errors, data_size):
    """Plot the misclassification error for each epoch in red.

    Parameters
    ----------
    classification_errors : list of int
        Absolute values of misclassified samples for each epoch.
    data_size : int
        Total number of training samples.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure = plt.figure()
    axes = figure.add_subplot(111)

    axes.plot(np.arange(len(classification_errors)),
              np.array(classification_errors) / data_size, 'r-')

    return figure

def section_3_1_1(filename, generate_and_save_data=False, show_plots=True):
    """Execute the instructions corresponding to section 3.1.1.

    Extract and plot data using extract_2_class_data() and plot_2_class_data().
    If generate_and_save_data==True, generate data and save it to a local file
    beforehand using generate_2_class_data() and
    shuffle_and_save_2_class_data(). The plots are not displayed if
    show_plots==False.

    Parameters
    ----------
    filename : sring
        Filename used to save or extract the data.
    generate_and_save_data : bool
    show_plots : bool

    Returns
    -------
    numpy.ndarray
        Training data described above.
    """
    if (generate_and_save_data):
        class_size = 100
        negative_mean = np.array([-1., 0.])
        positive_mean = np.array([1., 0.])
        means = [negative_mean, positive_mean]
        negative_covariance = np.mat([[0.2**2, 0.], [0., 1.]])
        positive_covariance = np.mat([[0.5**2, 0.], [0., 1.5]])
        covariances = [negative_covariance, positive_covariance]
        
        patterns = generate_2_class_data(means, covariances, class_size)
        shuffle_and_save_2_class_data(patterns, filename)
    
    data = extract_2_class_data(filename)
    plot = plot_2_class_data(data)
    if show_plots:
        plt.show(plot)
    return data

def section_3_1_2(data, step_size=0.001, batch=False, max_epochs=100,
                  export_animation=False, show_plots=True):
    """Execute the instructions correspondig to section 3.1.2.

    Train the network using data, step_size, batch and max_epochs; then, plot
    the resulting separating hyperplane together with the data and show the
    plot if show_plots==True; then, if export_animation==True, create a
    directory and export plots for each epoch of learning. The directory is
    created in the current folder and is named "animation-perceptron-3.1.2"
    together with a suffix to prevent overwriting existing directories.
    Parameters
    ----------
    data : numpy.ndarray
        Data extracted using extract_2_class_data().
    step_size : float
        Please see core.perceptron_learning().
    batch : bool
        Please see core.perceptron_learning().
    max_epochs : int
        Please see core.perceptron_learning().
    export_animation : bool
    show_plots : bool
    """
    network = core.initialize_weights([3, 1])
    weights_history, misclassification_history = core.perceptron_learning(
        network, data[:, :2].T, data[:, 2].T, step_size, batch, max_epochs)

    learning_plot = plot_classification_error(misclassification_history,
                                              data.shape[0])
    separated_data_plot = plot_2_class_data(data, weights_history[-1])
    if show_plots:
        plt.show(learning_plot)
        plt.show(separated_data_plot)

    if export_animation:
        i = 0
        dirname = "./animation-perceptron-3.1.2-" + str(i)
        while os.path.exists(dirname):
            i += 1
            dirname = "./animation-perceptron-3.1.2-" + str(i)
        os.makedirs(dirname, 0o755)

        for i, weights in enumerate(weights_history):
            current_figure = plot_2_class_data(data, weights)
            plt.figure(current_figure.number)
            plt.savefig(dirname + "/epoch-" + "%03d" % i + ".png")
            plt.close(current_figure)

filename = "part_1_linearly_separable.npy"
data = section_3_1_1(False, filename)
data = extract_2_class_data(filename)
section_3_1_2(data, step_size=0.001, batch=False, max_epochs=300,
              export_animation=True, show_plots=False)
