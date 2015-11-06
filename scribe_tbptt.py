import ipdb

import os
import math
import numpy
from numpy import array

import theano
from theano import tensor, function

from blocks.algorithms import ( GradientDescent,
    Adam, StepClipping, CompositeRule)

from blocks.bricks import (Tanh, MLP, Initializable,
                        Rectifier, Activation, Identity)

from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout)

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

import numpy as np
from theano import tensor as T
from play.utils import BivariateGMM
from cle.cle.utils import predict

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter
from blocks.extensions.saveload import Checkpoint
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot
from play.toy.segment_transformer import SegmentSequence
from model import BivariateGMMEmitter

floatX = theano.config.floatX

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

###################
# Define parameters of the model
###################

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'handwriting/')

batch_size = 20
frame_size = 3
k = 20
target_size = frame_size * k

hidden_size_recurrent = 400
readout_size =6*k+1

lr = 10 ** (5*numpy.random.rand() - 8)
exp_name = 'scribe_tbptt_{}'.format(lr)

dataset = Handwriting(('train',))
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
                #all batches the same size:
            batch_size*(dataset.num_examples/batch_size),
            batch_size))
data_stream = FilterSources(data_stream, 
                          sources = ('features',))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)
data_stream = SegmentSequence(data_stream, add_flag = True,
              seq_size = 100)
data_stream = FilterSources(data_stream, 
                          sources = ('features', 'features_mask','start_flag'))
data_stream = ForceFloatX(data_stream)

dataset = Handwriting(('valid',))
valid_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            batch_size*(dataset.num_examples/batch_size),
            batch_size))
valid_stream = FilterSources(valid_stream, 
                          sources = ('features',))
valid_stream = Padding(valid_stream)
valid_stream = Mapping(valid_stream, _transpose)
valid_stream = SegmentSequence(valid_stream, add_flag = True,
               seq_size = 12000)
valid_stream = FilterSources(valid_stream, 
                          sources = ('features', 'features_mask','start_flag'))
valid_stream = ForceFloatX(valid_stream)

x_tr = next(data_stream.get_epoch_iterator())

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')
start_flag = tensor.scalar('start_flag')

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
generator.initialize()

states = generator.transition.transition.apply.outputs

from blocks.utils import shared_floatx_zeros
states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
          for name in states}

cost_matrix = generator.cost_matrix(x, x_mask, **states)
cost = cost_matrix.sum()/x_mask.sum() + 0*start_flag
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
  batch_size = 8,
  iterate = True
  )[-2]

function([x, x_mask, start_flag], cost)(x_tr[0],x_tr[1], x_tr[2])
emit_fn = ComputationGraph(emit).get_theano_function()
emit_fn()

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

def make_regex(name_state = ""):
    return '[a-zA-Z_]*'+ name_state +'_final_value'

# Update the initial values with the last state
extra_updates = []
for name, var in states.items():
    update = T.switch(start_flag, 0.*var,
                VariableFilter(theano_name_regex=make_regex(name)
                    )(cg.auxiliary_variables)[0])
    extra_updates.append((var, update))

algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.), Adam(0.001)]))
algorithm.add_updates(extra_updates)

variables = [cost, min_sigma, max_sigma,
    min_mean, max_mean, mean_mean, mean_sigma, mean_corr,
    mean_data, sigma_data, min_corr, max_corr,
    max_data, min_data, mean_penup]

n_batches = 2000

train_monitor = TrainingDataMonitoring(
    variables=variables + [algorithm.total_step_norm,
    algorithm.total_gradient_norm],
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     variables,
     valid_stream,
     every_n_batches = n_batches,
     prefix="valid")

def _is_nan(log):
    #ipdb.set_trace()
    try:
      result = math.isnan(log.current_row['train_total_gradient_norm']) or \
               math.isnan(log.current_row['train_nll']) or \
               math.isnan(log.current_row['valid_nll']) or \
               math.isinf(log.current_row['train_total_gradient_norm']) or \
               math.isinf(log.current_row['train_nll']) or \
               math.isinf(log.current_row['valid_nll'])
      return result
    except:
      return False

extensions = extensions=[
    train_monitor,
    valid_monitor,
    Timing(every_n_batches = n_batches),
    Printing(every_n_batches = n_batches),
    Write(generator, every_n_batches = n_batches,
        save_name = save_dir + "samples/" + exp_name + ".png"),
    FinishAfter(before_epoch = False)
    .add_condition(["after_epoch"], _is_nan),
#    ProgressBar(),
    Checkpoint(save_dir + exp_name + ".pkl",after_epoch = True),
    SaveComputationGraph(emit),
    Plot(save_dir + "pkl/" + exp_name + ".png",
     [['train_nll',
       'valid_nll']],
     every_n_batches = 5*n_batches,
     email = False),
    Flush(every_n_batches = n_batches, after_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()
