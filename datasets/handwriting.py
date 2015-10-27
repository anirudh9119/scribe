import os
import ipdb

from fuel import config
from fuel.datasets import H5PYDataset

class Handwriting(H5PYDataset):
    filename = 'handwriting.hdf5'

    def __init__(self, which_sets, **kwargs):
        super(Handwriting, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], 'handwriting', self.filename)