from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

from utils import plot_hw
import ipdb

class Write(SimpleExtension):
    """Make your scribe write
    Parameters
    ----------
    steps : int Number of points to generate
    """
    def __init__(self, generator, steps=1000, n_samples = 5, 
            save_name = "sample_scribe.png", **kwargs):
        super(Write, self).__init__(**kwargs)
        emit = generator.generate(
          n_steps = steps,
          batch_size = n_samples,
          iterate = True
          )[-2]
        emit = emit.swapaxes(0,1)
        self.sample_fn = ComputationGraph(emit).get_theano_function()
        self.save_name = save_name

    def do(self, callback_name, *args):
        #ipdb.set_trace()
        plot_hw(self.sample_fn()[0], save_name = self.save_name)
