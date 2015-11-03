from blocks.bricks.sequence_generators import AbstractEmitter
from theano import tensor
from blocks.bricks import Initializable, Random
from blocks.bricks.base import application

from play.utils import BivariateGMM
from cle.cle.utils import predict

import theano
floatX = theano.config.floatX

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
        sigma = tensor.exp(sigma) + self.epsilon
        #sigma = tensor.nnet.softplus(sigma) + self.epsilon
        corr = tensor.tanh(corr)
        weight = tensor.nnet.softmax(weight)
        penup = tensor.nnet.sigmoid(penup)

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