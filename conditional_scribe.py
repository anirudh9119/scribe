import ipdb

import os
import math
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
from extensions import WriteConditional

import numpy as np
from theano import tensor as T
from play.utils import BivariateGMM
from cle.cle.utils import predict

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter
from blocks.extensions.saveload import Checkpoint
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot

floatX = theano.config.floatX

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

from model import BivariateGMMEmitter

###################
# Define parameters of the model
###################

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'handwriting/')

exp_name = 'conditional_scribe_0'

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

#data_stream = FilterSources(data_stream, 
#                          sources = ('features',))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)
#data_stream = ForceFloatX(data_stream)

dataset = Handwriting(('valid',))
valid_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, 10*batch_size))
valid_stream = Padding(valid_stream)
valid_stream = Mapping(valid_stream, _transpose)

x_tr = next(data_stream.get_epoch_iterator())

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')

context = tensor.imatrix('transcripts')
context_mask = tensor.matrix('transcripts_mask')

transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                   name = "gru_{}".format(i) ) for i in range(3)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

emitter = BivariateGMMEmitter(k = k)

source_names = [name for name in transition.apply.states if 'states' in name]

#68 characters
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.lookup import LookupTable

lookup = LookupTable(68,100)
embed = lookup.apply(context)

attention = SequenceContentAttention(
            state_names=source_names,
            attended_dim=100, #or is it 68
            match_dim=30,
            name="attention")

readout = Readout(
    readout_dim = readout_size,
    source_names =source_names + [attention.take_glimpses.outputs[0]],
    emitter=emitter,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              attention=attention,
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.001)
generator.push_initialization_config()

lookup.weights_init = IsotropicGaussian(0.01)
lookup.biases_init = Constant(0.001)
lookup.initialize()

#generator.transition.weights_init = initialization.Identity(0.98)
#generator.transition.biases_init = IsotropicGaussian(0.01,0.9)
generator.transition.push_initialization_config()
generator.initialize()

cost_matrix = generator.cost_matrix(x, x_mask, attended = embed,
                            attended_mask = context_mask)
cost = cost_matrix.sum(axis = 0).mean()
cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

transition_matrix = VariableFilter(
            theano_name_regex = "state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*np.eye(hidden_size_recurrent, dtype = floatX))

readouts = VariableFilter( applications = [generator.readout.readout],
    name_regex = "output")(cg.variables)[0]

mean, sigma, corr, weight, penup = emitter.components(readouts)

emit = generator.generate(
  n_steps = 400,
  iterate = True,
  attended = embed,
  attended_mask = context_mask,
  batch_size=embed.shape[1]
  )[-4]

function([x, x_mask, context, context_mask], cost)(x_tr[0],x_tr[1], x_tr[2], x_tr[3])
emit_fn = ComputationGraph(emit).get_theano_function()
emit_fn(x_tr[3], x_tr[2])[0].shape

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

parameters = cg.parameters

algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.), Adam(0.001)]))

variables = [cost, min_sigma, max_sigma,
    min_mean, max_mean, mean_mean, mean_sigma, mean_corr,
    mean_data, sigma_data, min_corr, max_corr,
    max_data, min_data, mean_penup]

n_batches = 10

train_monitor = TrainingDataMonitoring(
    variables=variables + [algorithm.total_step_norm,
    algorithm.total_gradient_norm],
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     variables,
     valid_stream,
     every_n_batches = 15*n_batches,
     prefix="valid")

def _is_nan(log):
    try:
      result = math.isnan(log.current_row['train_total_gradient_norm'])
      return result
    except:
      return False

extensions = extensions=[
    train_monitor,
    valid_monitor,
    Timing(every_n_batches = n_batches),
    Printing(every_n_batches = n_batches),
    WriteConditional(emit, every_n_batches = n_batches,
        save_name = save_dir + "samples/" + exp_name + ".png"),
    FinishAfter()
    .add_condition(["after_batch"], _is_nan),
    ProgressBar(),
    Checkpoint(save_dir + "pkl/" + exp_name + ".pkl",
        after_epoch = True),
    SaveComputationGraph(emit),
    Plot(save_dir + exp_name + ".png",
     [['train_nll',
       'valid_nll']],
     every_n_batches = 5*n_batches, email = False)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()

ipdb.set_trace()