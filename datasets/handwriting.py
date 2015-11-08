import os
import ipdb

from fuel import config
from fuel.datasets import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping, Padding, FilterSources, ForceFloatX

from play.toy.segment_transformer import SegmentSequence
from play.utils import _transpose

class Handwriting(H5PYDataset):
    filename = 'handwriting.hdf5'

    def __init__(self, which_sets, **kwargs):
        super(Handwriting, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], 'handwriting', self.filename)


def stream_handwriting(which_sets , batch_size,
					   seq_size, tbptt = True):
	dataset = Handwriting(which_sets)
	data_stream = DataStream.default_stream(
	            dataset, iteration_scheme=SequentialScheme(
	            batch_size*(dataset.num_examples/batch_size),
	            batch_size))
	data_stream = FilterSources(data_stream, 
	                          sources = ('features',))
	data_stream = Padding(data_stream)
	data_stream = Mapping(data_stream, _transpose)

	if tbptt:
		data_stream = SegmentSequence(data_stream, add_flag = True,
		               seq_size = seq_size)
	
	data_stream = ForceFloatX(data_stream)

	return data_stream

if __name__ == "__main__":
	import ipdb
	data_stream = stream_handwriting(('train',), 64, 100, True)
	x_tr = next(data_stream.get_epoch_iterator())
	#ipdb.set_trace()
