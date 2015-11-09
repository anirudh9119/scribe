import ipdb
import numpy

class ScribeParameters:
	def __init__(self, seed = 0):
		numpy.random.seed(seed = seed)
		self.batch_size = 20
		self.frame_size = 3
		self.k = 20
		self.target_size = self.frame_size * self.k
		self.hidden_size_recurrent = 400
		self.lr = 10 ** (7*numpy.random.rand() - 8)
		self.max_size = 1200
		self.seq_size = 100
		self.tbptt_flag = True
		self.exp_name = 'baseline_{}'.format(seed)
		self.n_batches = 2000
		self.tbptt_flag = True

if __name__ == "__main__":
	scribe_params = ScribeParameters()
	ipdb.set_trace()