import ipdb
import numpy

class ScribeParameters:
	def __init__(self):
		self.batch_size = 20
		self.frame_size = 3
		self.k = 20
		self.target_size = self.frame_size * self.k
		self.hidden_size_recurrent = 400
		self.readout_size =6*self.k+1
		self.lr = 3e-4
		self.lr = 10 ** (5*numpy.random.rand() - 8)
		self.exp_name = 'scribe_tbptt_{}'.format(self.lr)

if __name__ == "__main__":
	scribe_params = ScribeParameters()
	ipdb.set_trace()