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
    A list of numpy.ndarray of size 2. Each numpy.ndarray represents a class and
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

def plot_2_class_data(data):
    """Plot the data extracted by extract_2_class_data().

    Plot data using red 'x' markers for the negative class and blue '+' markers
    for the positive class.
    """
    negative_indices = data[:, 2] == -1
    positive_indices = data[:, 2] == 1
    plt.plot(data[negative_indices, 0], data[negative_indices, 1],
             linestyle="None", marker="x", markeredgecolor="r",
             markerfacecolor="r")
    plt.plot(data[positive_indices, 0], data[positive_indices, 1],
             linestyle="None", marker="+", markeredgecolor="b",
             markerfacecolor="b")
    plt.show()

def section_3_1_1(generate_and_save_data=False):
    """Execute the instructions corresponding to section 3.1.1.

    Extract and plot data using extract_2_class_data() and plot_2_class_data().
    If generate_and_save_data==True, generate data and save it to a local file
    beforehand using generate_2_class_data() and
    shuffle_and_save_2_class_data().

    Parameters
    ----------
    generate_and_save_data : bool
    """
    filename = "part_1_linearly_separable.npy"
    
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
    plot_2_class_data(data)

section_3_1_1(False)
