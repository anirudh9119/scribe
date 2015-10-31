import ipdb

import os
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

from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, AbstractEmitter)

from blocks.extensions import (FinishAfter, Printing,
                        SimpleExtension, Timing, ProgressBar)

from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                    DataStreamMonitoring)

from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks import initialization
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
from play.utils import BivariateGMM
from cle.cle.utils import predict

class BivariateGMMEmitter(AbstractEmitter, Initializable, Random):
    """A mixture of gaussians emitter for x,y and logistic for pen-up/down.
    Parameters
    ----------
    k : number of components
    """
    def __init__(self, k=20, epsilon=1e-5, **kwargs):
        self.k = k
        self.epsilon = epsilon
        super(BivariateGMMEmitter, self).__init__(**kwargs)

    def components(self, readouts):
        "Extract parameters of the distribution."
        k = self.k
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        #Reshaped
        mean = readouts[:, 0:2*k].reshape((-1,2,k))
        sigma = readouts[:, 2*k:4*k].reshape((-1,2,k))
        corr = readouts[:, 4*k:5*k]
        weight = readouts[:, 5*k:6*k]
        penup = readouts[:, 6*k:]

        #mean = mean
        sigma = T.exp(sigma) + self.epsilon
        #sigma = T.nnet.softplus(sigma) + self.epsilon
        corr = T.tanh(corr)
        weight = T.nnet.softmax(weight)
        penup = T.nnet.sigmoid(penup)

        mean.name = "mu"
        sigma.name = "sigma"
        corr.name = "corr"
        weight.name = "coeff"
        penup.name = "penup"

        return mean, sigma, corr, weight, penup

    @application
    def emit(self, readouts):
        """Sample from the distribution.
        """
        mu, sigma, corr, coeff, penup = self.components(readouts)
        
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ), axis=1)
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sigma = sigma[T.arange(sigma.shape[0]), :, idx]
        corr = corr[T.arange(corr.shape[0]), idx]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sigma_x = sigma[:,0]
        sigma_y = sigma[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        un = self.theano_rng.uniform(size=penup.shape)
        penup = T.cast(un < penup, floatX)

        s_x = (mu_x + sigma_x * z[:,0]).dimshuffle(0,'x')
        s_y = mu_y + sigma_y * ( (z[:,0] * corr) + (z[:,1] * T.sqrt(1.-corr**2)))
        s_y = s_y.dimshuffle(0,'x')
        s = T.concatenate([penup,s_x,s_y], axis = 1)

        return s

    @application
    def cost(self, readouts, outputs):
        """ Bivariate Gaussian NLL
        """
        mu, sigma, corr, coeff, penup = emitter.components(readouts)
        return BivariateGMM(outputs, mu, sigma, corr, coeff, penup)

    @application
    def initial_outputs(self, batch_size):
        return T.zeros((batch_size,3))

    def get_dim(self, name):
        if name == 'inputs':
            #(2k: mean, 2k: variance, k: corr, k: weights, 1:penup)
            return 6*self.k+1
        if name == 'outputs':
            return 3

        return super(BivariateGMMEmitter, self).get_dim(name)


###################
# Define parameters of the model
###################

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'handwriting/')

exp_name = 'scribe_1'

batch_size = 20
frame_size = 3
k = 20
target_size = frame_size * k

hidden_size_recurrent = 400
readout_size =6*k+1

lr = 3e-4

dataset = Handwriting(('train',))
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))
data_stream = FilterSources(data_stream, 
                          sources = ('features',))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)

dataset = Handwriting(('valid',))
valid_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, 10*batch_size))
valid_stream = FilterSources(valid_stream, 
                          sources = ('features',))
valid_stream = Padding(valid_stream)
valid_stream = Mapping(valid_stream, _transpose)
valid_stream = ForceFloatX(valid_stream)

x_tr = next(data_stream.get_epoch_iterator())

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')

transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                   name = "gru_{}".format(i) ) for i in range(3)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

emitter = BivariateGMMEmitter(k = k)

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
generator.biases_init = Constant(0.001)
generator.push_initialization_config()


#generator.transition.weights_init = initialization.Identity(0.98)
#generator.transition.biases_init = IsotropicGaussian(0.01,0.9)
generator.transition.push_initialization_config()
generator.initialize()

cost_matrix = generator.cost_matrix(x, x_mask)
cost = cost_matrix.sum(axis = 0).mean()
cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter
transition_matrix = VariableFilter(
            theano_name_regex = "state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*np.eye(hidden_size_recurrent, dtype = floatX))

readouts = VariableFilter( applications = [generator.readout.readout],
    name_regex = "output")(cg.variables)[0]

mean, sigma, corr, weight, penup = emitter.components(readouts)

cost_reg = cost + 0
#for weight in VariableFilter( roles = [WEIGHT])(cg.variables):
#    cost_reg += 0.01*(weight**2).sum()

#ipdb.set_trace()
#cost_reg += 0.01*sigma.mean() #+ 0.0001*(mean**2).mean()
#cost_reg += 0.01*(penup**2).mean()
cost_reg.name = "reg_cost"

emit = generator.generate(
  n_steps = 400,
  batch_size = 8,
  iterate = True
  )[-2]

cg = ComputationGraph(cost_reg)
model = Model(cost_reg)

min_sigma = sigma.min(axis=(0,2)).copy(name="sigma_min")
mean_sigma = sigma.mean(axis=(0,2)).copy(name="sigma_mean")
max_sigma = sigma.max(axis=(0,2)).copy(name="sigma_max")

min_mean = mean.min(axis=(0,2)).copy(name="mu_min")
mean_mean = mean.mean(axis=(0,2)).copy(name="mu_mean")
max_mean = mean.max(axis=(0,2)).copy(name="mu_max")

min_corr = corr.min().copy(name="corr_min")
mean_corr = corr.mean().copy(name="corr_mean")
max_corr = corr.max().copy(name="corr_max")

mean_data = x.mean(axis=(0,1)).copy(name="data_mean")
sigma_data = x.std(axis=(0,1)).copy(name="data_std")
max_data = x.max(axis=(0,1)).copy(name="data_max")
min_data = x.min(axis=(0,1)).copy(name="data_min")

mean_penup = penup.mean().copy(name="penup_mean")

#ipdb.set_trace()

function([x, x_mask], cost)(x_tr[0],x_tr[1])
emit_fn = ComputationGraph(emit).get_theano_function()
emit_fn()

parameters = cg.parameters

algorithm = GradientDescent(
    cost=cost_reg, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.), Adam(0.001)]))

variables = [cost, min_sigma, max_sigma,
    min_mean, max_mean, mean_mean, mean_sigma, mean_corr,
    mean_data, sigma_data, min_corr, max_corr, cost_reg,
    max_data, min_data, mean_penup]

n_batches = 100

train_monitor = TrainingDataMonitoring(
    variables=variables,
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     variables,
     valid_stream,
     after_epoch = True,
     every_n_batches = n_batches,
     prefix="valid")

from blocks.extensions.saveload import Checkpoint
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot

#def _is_nan(log):
#    return any(v != v for v in log.current_row.itervalues())

extensions = extensions=[
    train_monitor,
    valid_monitor,
    Timing(every_n_batches = n_batches),
    Printing(every_n_batches = n_batches),
    Write(generator, every_n_batches = n_batches,
        save_name = save_dir + "samples/" + exp_name + ".png"),
    FinishAfter(),
    # .add_condition(["after_batch"], _is_nan),
    ProgressBar(),
    Checkpoint(save_dir + exp_name + ".pkl",after_epoch = True),
    SaveComputationGraph(emit),
    Plot(save_dir + "pkl/" + exp_name + ".png",
     [['train_nll',
       'valid_nll']],
     every_n_batches = 5*n_batches)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()

ipdb.set_trace()