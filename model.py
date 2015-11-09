from blocks.bricks.sequence_generators import AbstractEmitter
from theano import tensor
from blocks.bricks import Initializable, Random
from blocks.bricks.base import application
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import SequenceGenerator, Readout
from blocks.filter import VariableFilter
from blocks.utils import shared_floatx_zeros
    
from cle.cle.utils import predict
from cle.cle.utils.op import logsumexp

import numpy
import theano
floatX = theano.config.floatX

def BivariateGMM(y, mu, sigma, corr, coeff, binary, epsilon = 1e-5):
    """
    Bivariate gaussian mixture model negative log-likelihood
    Parameters
    ----------
    """
    n_dim = y.ndim
    shape_y = y.shape
    y = y.reshape((-1, shape_y[-1]))
    y = y.dimshuffle(0, 1, 'x')

    mu_1 = mu[:,0,:]
    mu_2 = mu[:,1,:]

    sigma_1 = sigma[:,0,:]
    sigma_2 = sigma[:,1,:]

    binary = (binary+epsilon)*(1-2*epsilon)

    c_b =  tensor.sum( tensor.xlogx.xlogy0(y[:,0,:],  binary) +
              tensor.xlogx.xlogy0(1 - y[:,0,:], 1 - binary), axis = 1)

    inner1 =  (0.5*tensor.log(1.-corr**2 + epsilon)) + \
                         tensor.log(sigma_1) + tensor.log(sigma_2) +\
                         tensor.log(2. * numpy.pi)

    Z = (((y[:,1,:] - mu_1)/sigma_1)**2) + (((y[:,2,:] - mu_2) / sigma_2)**2) - \
        (2. * (corr * (y[:,1,:] - mu_1)*(y[:,2,:] - mu_2)) / (sigma_1 * sigma_2))
    inner2 = 0.5 * (1. / (1. - corr**2 + epsilon))
    cost = - (inner1 + (inner2 * Z))

    nll = -logsumexp(tensor.log(coeff) + cost, axis=1) - c_b
    return nll.reshape(shape_y[:-1], ndim = n_dim-1)

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

    @application(outputs=["mu", "sigma", "corr", "coeff", "penup"])
    def components(self, readouts):
        "Extract parameters of the distribution."
        k = self.k
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        #Reshaped
        mu = readouts[:, 0:2*k].reshape((-1,2,k))
        sigma = readouts[:, 2*k:4*k].reshape((-1,2,k))
        corr = readouts[:, 4*k:5*k]
        weight = readouts[:, 5*k:6*k]
        penup = readouts[:, 6*k:]

        #mu = mu
        #sigma = tensor.exp(sigma) + self.epsilon
        sigma = tensor.nnet.softplus(sigma) + self.epsilon
        corr = tensor.tanh(corr)
        weight = tensor.nnet.softmax(weight) + self.epsilon
        penup = tensor.nnet.sigmoid(penup)

        mu.name = "mu"
        sigma.name = "sigma"
        corr.name = "corr"
        weight.name = "coeff"
        penup.name = "penup"

        return mu, sigma, corr, weight, penup

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

        mu = mu[tensor.arange(mu.shape[0]), :, idx]
        sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
        corr = corr[tensor.arange(corr.shape[0]), idx]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sigma_x = sigma[:,0]
        sigma_y = sigma[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        un = self.theano_rng.uniform(size=penup.shape)
        penup = tensor.cast(un < penup, floatX)

        s_x = (mu_x + sigma_x * z[:,0]).dimshuffle(0,'x')
        s_y = mu_y + sigma_y * ( (z[:,0] * corr) + (z[:,1] * tensor.sqrt(1.-corr**2)))
        s_y = s_y.dimshuffle(0,'x')
        s = tensor.concatenate([penup,s_x,s_y], axis = 1)

        return s

    @application
    def cost(self, readouts, outputs):
        """ Bivariate Gaussian NLL
        """
        mu, sigma, corr, coeff, penup = self.components(readouts)
        return BivariateGMM(outputs, mu, sigma, corr, coeff, penup, self.epsilon)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size,3))

    def get_dim(self, name):
        if name == 'inputs':
            #(2k: mean, 2k: variance, k: corr, k: weights, 1:penup)
            return 6*self.k+1
        if name == 'outputs':
            return 3

        return super(BivariateGMMEmitter, self).get_dim(name)

class Scribe(Initializable):
    """Scribe is here to write for you!
    You will not need another pencil again.
    """
    def __init__(self,hidden_size_recurrent, k, **kwargs):
        super(Scribe, self).__init__(**kwargs)

        readout_size =6*k+1
        transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                      name = "gru_{}".format(i) ) for i in range(3)]

        transition = RecurrentStack( transition,
                name="transition", skip_connections = True)

        emitter = BivariateGMMEmitter(k = k)

        source_names = [name for name in transition.apply.states 
                                      if 'states' in name]

        readout = Readout(
            readout_dim = readout_size,
            source_names =source_names,
            emitter=emitter,
            name="readout")

        self.generator = SequenceGenerator(readout=readout, 
                                  transition=transition,
                                  name = "generator")

        self.children = [self.generator]

    def monitoring_vars(self, cg):
        readout = self.generator.readout
        readouts = VariableFilter( applications = [readout.readout],
            name_regex = "output")(cg.variables)[0]

        mean, sigma, corr, weight, penup = readout.emitter.components(readouts)

        min_sigma = sigma.min(axis=(0,2)).copy(name="sigma_min")
        mean_sigma = sigma.mean(axis=(0,2)).copy(name="sigma_mean")
        max_sigma = sigma.max(axis=(0,2)).copy(name="sigma_max")

        min_mean = mean.min(axis=(0,2)).copy(name="mu_min")
        mean_mean = mean.mean(axis=(0,2)).copy(name="mu_mean")
        max_mean = mean.max(axis=(0,2)).copy(name="mu_max")

        min_corr = corr.min().copy(name="corr_min")
        mean_corr = corr.mean().copy(name="corr_mean")
        max_corr = corr.max().copy(name="corr_max")

        mean_penup = penup.mean().copy(name="penup_mean")

        monitoring_vars = [mean_sigma, min_sigma,
            min_mean, max_mean, mean_mean, max_sigma,
            mean_corr, min_corr, max_corr, mean_penup]

        return monitoring_vars

