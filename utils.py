from fuel.datasets.hdf5 import H5PYDataset
from datasets.handwriting import Handwriting
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import ipdb

train = Handwriting(('all',))
handle = train.open()

x_tr = train.get_data(handle, slice(89, 97))

# Standardize substract first point
X, TRANSCRIPTS = x_tr

#ipdb.set_trace()

def plot_hw(X, ax=None):
    if ax is None:
        f, ax = pyplot.subplots()

    cuts = np.where(X[:, 0] == 1)[0]
    start = 0
    for cut_value in cuts:
        ax.plot(X[start:cut_value, 1], X[start:cut_value, 2],
                 'k-', linewidth=1.5)
        start = cut_value+1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def plot_H(X, save_name=None):
    n_samples = len(X)
    f, axarr = pyplot.subplots(n_samples, sharex=True)

    for i in range(n_samples):
        plot_hw(X[i], ax=axarr[i])

    if save_name is None:
        pyplot.show()
    else:
        pyplot.savefig(save_name)


all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('A') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', ' ', '"', '<UNK>', "'"])

code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'

for transcript in TRANSCRIPTS:
    print "".join([code2char[x] for x in transcript])

plot_H(X, save_name = 'test.png')

#ipdb.set_trace()
