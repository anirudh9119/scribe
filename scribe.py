import ipdb

import numpy
from numpy import array

import theano
from theano import tensor, function

from blocks.algorithms import ( GradientDescent,
    Adam, StepClipping, CompositeRule)
from blocks.bricks import Initializable, Random
from blocks.bricks import (Tanh, MLP, Initializable,
                        Rectifier, Activation, Identity)
from blocks.bricks.base import application

from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, AbstractEmitter)

from blocks.extensions import FinishAfter, Printing, SimpleExtension

from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                    DataStreamMonitoring)

from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation

from fuel.transformers import Mapping, Padding, FilterSources, ForceFloatX
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from datasets.handwriting import Handwriting
from extensions import Write

floatX = theano.config.floatX

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

import numpy as np
from theano import tensor as T

class SketchEmitter(AbstractEmitter, Initializable, Random):
    """A mixture of gaussians emitter for x,y and logistic for pen-up/down.

    Code from udibr.
    Parameters
    ----------
    k : number of components
    """
    def __init__(self, k=20, epsilon=1e-5, **kwargs):
        self.k = k
        self.epsilon = epsilon
        super(SketchEmitter, self).__init__(**kwargs)

    def components(self, readouts):
        "Extract parameters of the distribution."
        k = self.k
        output_norm = 2*k
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        mean = readouts[:, 0:2*k].reshape((-1,k,2))
        sigma = readouts[:, 2*k:4*k].reshape((-1,k,2))
        corr = readouts[:, 4*k:5*k]
        weight = readouts[:, 5*k:6*k]
        penup = readouts[:, 6*k:]

        #mean = mean
        sigma = T.exp(sigma) + self.epsilon
        #sigma = T.nnet.softplus(sigma) + self.epsilon
        corr = T.tanh(corr)
        weight = T.nnet.softmax(weight)
        penup = T.nnet.sigmoid(penup)

        return mean, sigma, corr, weight, penup

    @application
    def emit(self, readouts):
        """Sample from the distribution.

        """
        mean, sigma, corr, weight, penup = self.components(readouts)
        k = self.k
        batch_size = readouts.shape[0]

        nr = self.theano_rng.normal(
            size=(batch_size, k, 2),
            avg=0., std=1.)

        c = (1 - T.sqrt(1-corr**2))/(corr + self.epsilon)
        c = c.dimshuffle((0, 1, 'x'))
        nr = nr + nr[:,:,::-1]*c

        nr = nr / T.sqrt(1+c**2)
        nr = nr * sigma
        nr = nr + mean

        weight = self.theano_rng.multinomial(pvals=weight, dtype=floatX)
        xy = nr * weight[:,:,None]
        xy = xy.sum(axis=1)
        un = self.theano_rng.uniform(size=(batch_size, 1))
        penup = T.cast(un < penup, floatX)
        res = T.concatenate([penup, xy], axis=1)

        return res

    @application
    def cost(self, readouts, outputs):
        """ Bivariate Gaussian NLL
        """
        nll_ndim = readouts.ndim - 1
        nll_shape = readouts.shape[:-1]
        outputs = outputs.reshape((-1,3))
        mean, sigma, corr, weight, penup = self.components(readouts)
        d = outputs[:,1:].dimshuffle((0, 'x', 1)) - mean
        sigma2 = sigma[:,:,0] * sigma[:,:,1] + self.epsilon
        z = d ** 2 / sigma ** 2
        z = z.sum(axis=-1) - 2 * corr * (d[:,:,0] * d[:,:,1]) / sigma2
        corr1 = 1 - corr ** 2 + self.epsilon
        n = - z / (2 * corr1)
        nmax = n.max(axis=-1, keepdims=True)
        n = n - nmax
        n = T.exp(n) / (2*np.pi*sigma2*T.sqrt(corr1))
        nll = -T.log((n * weight).sum(axis=-1, keepdims=True) + self.epsilon)
        nll -= nmax
        # Change this 100!!!!
        nll += 100*T.nnet.binary_crossentropy(penup, outputs[:,:1])

        return nll.reshape(nll_shape, ndim=nll_ndim)

    @application
    def initial_outputs(self, batch_size):
        return T.zeros((batch_size,3))

    def get_dim(self, name):
        if name == 'inputs':
            #(2k: mean, 2k: variance, k: corr, k: weights, 1:penup)
            return 6*self.k+1
        if name == 'outputs':
            return 3

        return super(SketchEmitter, self).get_dim(name)

###################
# Define parameters of the model
###################

batch_size = 10 #64 for tpbtt
frame_size = 3
k = 20
target_size = frame_size * k

depth_x = 2
hidden_size_mlp_x = 20

depth_theta = 2
hidden_size_mlp_theta = 20
hidden_size_recurrent = 20
readout_size =6*k+1

lr = 3e-4

dataset = Handwriting(('all',))
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))
data_stream = FilterSources(data_stream, 
                          sources = ('features',))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)

x_tr = next(data_stream.get_epoch_iterator())

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')

transition = [LSTM(dim=hidden_size_recurrent, 
                   name = "lstm_{}".format(i) ) for i in range(3)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

emitter = SketchEmitter(k = k)

source_names = [name for name in transition.apply.states if 'states' in name]
readout = Readout(
    readout_dim = readout_size,
    source_names =source_names,
    emitter=emitter,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.push_initialization_config()

generator.transition.biases_init = IsotropicGaussian(0.01,1)
generator.transition.push_initialization_config()
generator.initialize()

cost_matrix = generator.cost_matrix(x, x_mask)
cost = cost_matrix.sum()/x_mask.sum()
cost.name = "sequence_log_likelihood"

cg = ComputationGraph(cost)
model = Model(cost)

emit = generator.generate(
  n_steps = 10,
  batch_size = 8,
  iterate = True
  )[-2]

#ipdb.set_trace()

function([x, x_mask], cost)(x_tr[0],x_tr[1])
emit_fn = ComputationGraph(emit).get_theano_function()
emit_fn()

parameters = cg.parameters

algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(0.005)]))

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches = 10,
    prefix="train")

extensions = extensions=[
    train_monitor,
    Printing(every_n_batches = 10),
    Write(generator,every_n_batches = 50),
    FinishAfter(after_n_batches = 100)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()

ipdb.set_trace()