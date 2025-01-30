"""
Variational Fractional Autoencoder (VFAE)
by Kian Ming A. Chai (cha847@csiro.au)
based heavily on the code by Ruthotto and Haber (2021) at https://github.com/EmoryMLIP/DeepGenerativeModelingIntro
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.distributions.continuous_bernoulli as LH

class VFAE(nn.Module):
    """
    Dimensions order: MCMC samples, Data-batch-size, per-data dimensions (1x28x28)
    """
    def __init__(self,e, g, gamma):
        """
        Initialize VFAE
        :param e: encoder network e(z|x), provides parameters of approximate posteriors
        :param g: generator g(z), provides samples in data space
        """
        super().__init__()
        self.e = e
        self.g = g
        self.gamma = gamma
        
    def encode(self, x):
        return self.e(x)    

    def generate_with(self, mu, logvar, Nz):
        z, eps = self.sample_ezx(mu, logvar, Nz)     # Nz samples per (mu, logvar)
        gz = self.g(z.view( (-1, mu.shape[-1] ) ) )  # collapse all dimensions except the latent dimension
        return gz, z, eps

    def ELBO(self, x, Nsamples):
        """
        Empirical lower bound on p_{\theta}(x)
        :param x: sample from dataset
        :param n: number  of MCMC sample. For stochastic encoder, this would be a tuple
        :return: metrics averged by both number of examples and number of MCMC samples
        """
        zparams = self.e(x)  # parameters of approximate posterior        
        return self.ELBO_at(zparams, x, Nsamples)

    def ELBOscores(self, x, Nsamples):
        elbo, data_score, prior_score, *_ = self.ELBO(x, Nsamples)
        return elbo, data_score, prior_score
        
    def ELBO_at(self, zparams, x, Ns):                
        N = x.shape[0] # number of data samples
        
        # flatten all parameters
        mu, logvar = zparams
        mu = mu.view(-1, mu.shape[-1])
        logvar = logvar.view(-1, logvar.shape[-1])

        Nu = mu.shape[0] // N
        Nz = Ns // Nu # in case we have more than one parameter per sample
        
        # z is generated from eps. Conversely, eps is standardized z.
        gz, z, eps = self.generate_with(mu, logvar, Nz)
        gz = gz.view( (Ns, N) + x.shape[-3:] )        # torch vision has 3 dimensions per sample: C*H*W
        
        # Get all the required (log) densities
        log_px = self.log_prob_px(x,gz)               # data    
        log_pz = self.log_prob_pz(z).view(Ns, -1)     # prior
        log_qzx = self.log_prob_qzx_eps(eps, logvar).view(Ns, -1) # fractional posterior

        if self.gamma >= 1.0: # conventional ELBO (gamma=1.0) or beta-VAE (gamma > 1.0)
            sZd = torch.sum(log_px) / Ns;    
            sZc = torch.sum(log_qzx - log_pz) * (self.gamma / Ns);
        else:
            ngma = 1.0 - self.gamma;
            rgma = ngma / self.gamma;
            sZd = ( torch.sum(torch.logsumexp( ngma * log_px, dim=0)) - N * np.log(Ns) ) / ngma;
            sZc = ( torch.sum(torch.logsumexp( rgma * (log_qzx - log_pz), dim=0)) - N * np.log(Ns) ) / rgma;
        
        return (-sZd+sZc)/N, (-sZd/N).item(), (sZc/N).item(), gz.detach(), mu.detach()

    def sample_x(self, z):
        gz = self.g(z);
        likelihood = LH.ContinuousBernoulli(probs = gz)
        x = likelihood.sample();
        return x;

    def mean_x(self, z):
        gz = self.g(z);
        likelihood = LH.ContinuousBernoulli(probs = gz)
        x = likelihood.mean;
        return x;
    
    def sample_ezx(self, mu, logvar, n=0):
        """
        draw sample from approximate posterior

        :param mu: mean of approximate posterior (optional; will be computed here if is None)
        :param logvar: log-variance of approximate posterior (optional; will be computed here if is None)
        :param n: number of samples or return the mean (if 0)
        :return:
        """
        if n > 0:
            shapen = (n,) + logvar.shape;
            std = torch.exp(logvar)
            eps = torch.randn(shapen, device=std.get_device())
            return std.unsqueeze(0).expand(shapen) * eps + mu.unsqueeze(0).expand(shapen), eps
        else:
            return mu, torch.zeros(logvar.shape) # Only one sample, so we don't expand dimensions for now

    def log_prob_qzx_eps(self, eps, logvar):
        """
        :return: log(q(z|x))
        """
        log_qzx = -0.5*torch.norm(eps, dim=-1)**2 - 0.5*torch.sum(logvar, dim=-1) - (eps.shape[-1]/2)*np.log(2*np.pi)        
        return log_qzx
    
    def log_prob_ezx(self,z,x):
        """
        :param z: latent sample
        :param x: data sample
        :return: log(e(z|x))
        """
        mu, logvar = self.encode(x)
        ezx = -torch.sum((0.5 / torch.exp(logvar)) * (z - mu) ** 2, dim=-1) - 0.5 * torch.sum(logvar,dim=-1) - (z.shape[-1]/2)*np.log(2*np.pi)
        return ezx

    def log_prob_px(self,x,gz):
        """
        :param z: latent sample
        :param x: data sample
        :return: log(p(x|z))
        """
        likelihood = LH.ContinuousBernoulli(probs = gz)
        if gz.dim() == x.dim():            
            px = likelihood.log_prob(x);
        else: # Expand to MCMC samples
            px = likelihood.log_prob(x.unsqueeze(0).expand(gz.shape));
        px = torch.sum(px,dim=(-3, -2, -1)) # torch vision has 3 dimensions per sample: C*H*W
        return px

    def log_prob_pz(self,z):
        """
        :param z: latent sample
        :return: log(p(z), log prior, which is multivariate normal
        """
        n = z.shape[-1]  # cha847@csiro.au: there was a bug in the original code with the Ruthotto and Haber paper
        pz = - 0.5 * torch.norm(z, dim=-1) ** 2  - (n/2)*np.log(2*np.pi)
        return pz

    def log_prob_pzx(self,z,x,gz):
        """
        :param z: latent sample
        :param x: data sample
        :return: log(p(z,x)) = log(p(x|z)) + log(p(z)), log(p(x|z|), log(p(z)) # the model distributions
        """
        px = self.log_prob_px(x, gz)
        pz = self.log_prob_pz(z)
        return px + pz, px, pz


# specialisation for semi-implicit, more work
class VFAESI(VFAE):
    def __init__(self,e, g, gamma):
        super().__init__(e, g, gamma)
        
    #@override
    def encode(self, x):
        qdist, u = self.e(x)
        return qdist[0], qdist[1]

    #@override
    def ELBO(self, x, Nsamples):
        """
        Empirical lower bound on p_{\theta}(x)
        :param x: sample from dataset
        :param n: number  of MCMC sample. For stochastic encoder, this would be a tuple
        :return: metrics averged by both number of examples and number of MCMC samples
        """
        self.e.nMCMC = Nsamples[1]            
        qdist, u = self.e(x)  # parameters of approximate posterior        
        return self.ELBO_at(qdist, x, Nsamples[0] ) + tuple([u.detach()])

    def ELBO_and_implicit(self, x, Nsamples):
        elbo, _, _, _, _, u = self.ELBO(x, Nsamples)
        return elbo, u
    


# Fractional and Bayes posterior together, even more work!
class VFBAESI(VFAESI):
    def __init__(self,e, g, gamma):
        super().__init__(e, g, gamma)

    #@override
    def encode(self, x):
        qdist, rdist, u = self.e(x)
        return rdist[0], rdist[1]

    def ELBO(self, x, Nsamples=1):
        """
        Empirical lower bound on p_{\theta}(x)
        :param x: sample from dataset
        :param n: number  of MCMC sample. For stochastic encoder, this would be a tuple
        :param forceKL: use ELBO (KL divergence) despite gamma
        :return: metrics averged by both number of examples and number of MCMC samples
        """
        self.e.nMCMC = Nsamples[1]
        qdist, rdist, u = self.e(x)  # parameters of approximate posterior
        return self.ELBO_at(qdist, rdist, x, Nsamples[0], Nsamples[2] ) + tuple([u.detach()])

    def ELBOscores(self, x, Nsamples):
        elbo, data_score, cross_score, prior_score, *_ = self.ELBO(x, Nsamples)
        return elbo, data_score, cross_score, prior_score

    #@override
    def ELBO_and_implicit(self, x, Nsamples):
        elbo, _, _, _, _, _, u = self.ELBO(x, Nsamples)
        return elbo, u
            
    def ELBO_at(self, qdist, rdist, x, Ns, Nv):
        N = x.shape[0] # number of data samples
        
        # no-op for static encoder
        qmu, qlogvar = qdist
        rmu, rlogvar = rdist
        qmu = qmu.view(-1, qmu.shape[-1])
        qlogvar = qlogvar.view(-1, qlogvar.shape[-1])
        rmu = rmu.view(-1, rmu.shape[-1])
        rlogvar = rlogvar.view(-1, rlogvar.shape[-1])

        Nu = rmu.shape[0] // N
        Nz = Ns // Nu # in case we have more than one parameter per sample
        
        # z is generated from eps. Conversely, eps is standardized z.
        gz, rz, reps = self.generate_with(rmu, rlogvar, Nz)
        gz = gz.view( (Ns, N) + x.shape[-3:] );  # torch vision has 3 dimensions per sample: C*H*W
        qz, qeps = self.sample_ezx(qmu, qlogvar, Nz) # z's, (u's, data), q-dim
        
        # Get all the required (log) densities
        log_px = self.log_prob_px(x, gz)             # data
        log_pz = self.log_prob_pz(qz).view(Ns, -1)   # prior
        log_qzx = self.log_prob_qzx_eps(qeps, qlogvar).view(Ns, -1) # fractional posterior
        log_rzx = self.log_prob_qzx_eps(reps, rlogvar).view(Ns, -1) # Bayes posterior 

        # select the implicits for cross evaluation
        qXmu, qXlogvar = self.sample_q_for_r(Nv, Nu, N, qmu, qlogvar)
        log_qXzx = self.log_prob_qzx_rz(rz.view(Nz,Nu,N,-1), Nv, qXmu, qXlogvar).view(Ns, -1) # Bayes-Fractional cross
             
        ngma = 1.0 - self.gamma;
        rgma = ngma / self.gamma;
        sZd = torch.sum(log_px) / Ns # data term
        sKL = torch.sum(log_rzx - log_qXzx) / Ns # cross-term
        sZc = ( torch.sum(torch.logsumexp( rgma * (log_qzx - log_pz), dim=0)) - N * np.log(Ns) ) / rgma; # regularisation term
        
        return (-sZd+sKL+sZc)/N, (-sZd/N).item(), (sKL/N).item(), (sZc/N).item(), gz.detach(), rmu.detach()
    
    def sample_q_for_r(self, Nv, Nu, N, qmu, qlogvar):
        # - Choose Nv out of Nr < Nu-1, so that we can exclude "self" in the selection
        # - can be replaced by randint if you are confident of probability of repeats is not zero in the model
        selrnd = torch.arange(0, Nv).unsqueeze(-1).unsqueeze(-1).tile(1, Nu, N)
        selex = torch.arange(1, Nu + 1).unsqueeze(1).tile(Nv, 1, N) # exclude self
        seloff = torch.arange(0, N).unsqueeze(0).unsqueeze(0).tile(Nv, Nu,1) # offset for linear indices
        sel = ( ((selrnd + selex) % Nu) * N + seloff).flatten()
        
        qXmu = qmu[sel,:].reshape(Nv, Nu, N, -1)
        qXlogvar = qlogvar[sel,:].reshape(Nv, Nu, N, -1)
        
        return qXmu, qXlogvar

    def log_prob_qzx_rz(self, z, Nv, mu, logvar):            
        Nv = mu.shape[0]        
        z = z.unsqueeze(0)
        mu = mu.unsqueeze(1)
        logvar = logvar.unsqueeze(1)
        
        logp = -0.5/Nv * torch.sum( (z - mu) ** 2 / torch.exp(logvar) - logvar, dim=(0,-1)) - (z.shape[-1]/2)*np.log(2*np.pi)

        return logp;

 
