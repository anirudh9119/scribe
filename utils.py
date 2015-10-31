import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import ipdb
import os

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path,'handwriting/')
std_values = np.load(os.path.join(data_path,'handwriting_std.npz'))
data_mean = std_values['data_mean']
data_std = std_values['data_std']

def plot_single(X, ax=None):
    # Plot a single example.
    if ax is None:
        f, ax = pyplot.subplots()

    X = data_std*X + data_mean

    x = np.cumsum(X[:,1])
    y = np.cumsum(X[:,2])

    cuts = np.where(X[:, 0] == 1)[0]
    start = 0
    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                 'k-', linewidth=1.5)
        start = cut_value+1
    ax.axis('equal')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)

def plot_hw(X, save_name=None):
    # Plot several examples in the same figure.
    n_samples = len(X)
    f, axarr = pyplot.subplots(n_samples, sharex=True)

    for i in range(n_samples):
        plot_single(X[i], ax=axarr[i])

    if save_name is None:
        pyplot.show()
    else:
        pyplot.savefig(save_name)

    pyplot.close()

if __name__ == '__main__':

    from datasets.handwriting import Handwriting
    all_chars = ([chr(ord('a') + i) for i in range(26)] +
                 [chr(ord('A') + i) for i in range(26)] +
                 [chr(ord('0') + i) for i in range(10)] +
                 [',', '.', ' ', '"', '<UNK>', "'"])

    code2char = dict(enumerate(all_chars))
    char2code = {v: k for k, v in code2char.items()}
    unk_char = '<UNK>'

    train = Handwriting(('train',))
    handle = train.open()

    x_tr = train.get_data(handle, slice(0,7))

    # Standardize substract first point
    X, TRANSCRIPTS = x_tr

    for transcript in TRANSCRIPTS:
        print "".join([code2char[x] for x in transcript])

    plot_hw(X, save_name = 'test.png')

