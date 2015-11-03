from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

from utils import plot_hw
import ipdb
import numpy

class Write(SimpleExtension):
    """Make your scribe write
    Parameters
    ----------
    steps : int Number of points to generate
    """
    def __init__(self, generator, steps=1000, n_samples = 2, 
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

class WriteConditional(SimpleExtension):
    """Make your scribe write
    Parameters
    ----------
    steps : int Number of points to generate
    """
    def __init__(self, emit, steps=1000, n_samples = 2, 
            save_name = "sample_scribe.png", **kwargs):
        super(WriteConditional, self).__init__(**kwargs)
        # emit = generator.generate(
        #   n_steps = steps,
        #   batch_size = n_samples,
        #   iterate = True
        #   )[-2]
        emit = emit.swapaxes(0,1)
        self.sample_fn = ComputationGraph(emit).get_theano_function()
        self.save_name = save_name
        phrase = "hello my friend"

        all_chars = ([chr(ord('a') + i) for i in range(26)] +
                     [chr(ord('A') + i) for i in range(26)] +
                     [chr(ord('0') + i) for i in range(10)] +
                     [',', '.', ' ', '"', '<UNK>', "'"])

        code2char = dict(enumerate(all_chars))
        char2code = {v: k for k, v in code2char.items()}
        unk_char = '<UNK>'

        phrase = [char2code[x] for x in phrase]
        phrase = numpy.array(phrase, dtype = 'int32')

        phrase.shape = phrase.shape + (1,)
        self.phrase = numpy.repeat(phrase, n_samples, 1)
        self.phrase_mask = numpy.ones(self.phrase.shape, dtype = 'float32')

    def do(self, callback_name, *args):
        #ipdb.set_trace()
        plot_hw(self.sample_fn(self.phrase_mask, self.phrase)[0], 
          save_name = self.save_name)
