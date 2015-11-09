import ipdb

import os
import sys
import math
import numpy
from numpy import array

import theano
from theano import tensor, function

from blocks.algorithms import ( GradientDescent,
    Adam, StepClipping, CompositeRule, RemoveNotFinite)

from blocks.extensions import (FinishAfter, Printing,
                        SimpleExtension, Timing)

from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                    DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.training import TrackTheBest

from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant

from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation

from datasets.handwriting import stream_handwriting
from extensions import Write

import numpy as np
from theano import tensor as T
from cle.cle.utils import predict

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter
from blocks.extensions.saveload import Checkpoint
from blocks.utils import shared_floatx
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot
from play.utils import regex_final_value, _transpose

from model import Scribe
from config import ScribeParameters

floatX = theano.config.floatX

###################
# Define parameters of the model
###################

save_dir = os.environ['RESULTS_DIR']
if 'handwriting' not in save_dir:
  save_dir = os.path.join(save_dir,'handwriting/')

if len(sys.argv) > 1:
  num_job = int(sys.argv[1])
else:
  num_job = 0

print num_job

exp_params = ScribeParameters(seed = num_job)
batch_size = exp_params.batch_size
k = exp_params.k
hidden_size_recurrent = exp_params.hidden_size_recurrent
readout_size =6*k+1

lr = exp_params.lr
exp_name = exp_params.exp_name
max_size = exp_params.max_size
seq_size = exp_params.seq_size
n_batches = exp_params.n_batches
tbptt_flag = exp_params.tbptt_flag

print "learning rate: ", lr

lr = shared_floatx(lr, "learning_rate")

if tbptt_flag:
  train_size = seq_size
else:
  train_size = max_size

data_stream = stream_handwriting(('train',), batch_size, train_size, tbptt_flag)
valid_stream = stream_handwriting(('valid',), batch_size, max_size, tbptt_flag)
x_tr = next(data_stream.get_epoch_iterator())

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')
start_flag = tensor.scalar('start_flag')

scribe = Scribe(hidden_size_recurrent,k)
scribe.weights_init = IsotropicGaussian(0.01)
scribe.biases_init = Constant(0.001)
scribe.initialize()

generator = scribe.generator
emitter = generator.readout.emitter

states = generator.transition.transition.apply.outputs

from blocks.utils import shared_floatx_zeros
states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
          for name in states}

cost_matrix = generator.cost_matrix(x, x_mask, **states)
cost = cost_matrix.sum()/x_mask.sum()

# Hack to incorporate the start flag in the computation graph.
if tbptt_flag:
  cost += 0*start_flag

cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

monitoring_vars = scribe.monitoring_vars(cg)

#Regularize variance

reg_cost = cost + 0. #+ 0.01*monitoring_vars[0].mean()
reg_cost.name = "reg_nll"

cg = ComputationGraph(reg_cost)
model = Model(reg_cost)

transition_matrix = VariableFilter(
            theano_name_regex = "state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*np.eye(hidden_size_recurrent, dtype = floatX))

emit = generator.generate(
  n_steps = 400,
  batch_size = 8,
  iterate = True
  )[-2]

# function([x, x_mask, start_flag], cost)(x_tr[0],x_tr[1], x_tr[2])
# emit_fn = ComputationGraph(emit).get_theano_function()
# emit_fn()

parameters = cg.parameters

# Update the initial values with the last state
extra_updates = []

if tbptt_flag:
  for name, var in states.items():
      update = T.switch(start_flag, 0.*var,
                  VariableFilter(theano_name_regex=regex_final_value(name)
                      )(cg.auxiliary_variables)[0])
      extra_updates.append((var, update))

algorithm = GradientDescent(
    cost=reg_cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.), Adam(lr), RemoveNotFinite()]))
algorithm.add_updates(extra_updates)

mean_data = x.mean(axis=(0,1)).copy(name="data_mean")
sigma_data = x.std(axis=(0,1)).copy(name="data_std")
max_data = x.max(axis=(0,1)).copy(name="data_max")
min_data = x.min(axis=(0,1)).copy(name="data_min")

variables = [reg_cost, cost, mean_data, sigma_data, max_data, min_data] + monitoring_vars

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

from extensions import LearningRateSchedule
from utils import _is_nan

extensions = extensions=[
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches = n_batches),
    Timing(every_n_batches = n_batches),
    Printing(every_n_batches = n_batches),
    Write(generator, save_name = save_dir + "samples/" + exp_name + ".png")
    .add_condition(["after_batch"],
      predicate=OnLogRecord('valid_nll_best_so_far')),
    FinishAfter()
    .add_condition(["after_batch"], _is_nan),
    #Checkpoint(save_dir + "pkl/" + exp_name + ".pkl",after_epoch = True),
    Checkpoint(save_dir + "pkl/best_" + exp_name + ".pkl",
      before_epoch = True)
    .add_condition(["after_batch"],
      predicate=OnLogRecord('valid_nll_best_so_far')),
    SaveComputationGraph(emit),
    Plot(save_dir + "progress/" + exp_name + ".png",
     [['train_nll',
       'valid_nll']],
     every_n_batches = 5*n_batches,
     email = False),
    Flush(every_n_batches = n_batches, after_epoch = True),
    LearningRateSchedule(
      lr, 
      'valid_nll',
      path = save_dir + "pkl/best_" + exp_name + ".pkl",
      every_n_batches = n_batches)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()

