# Distributions for the experiments for image generation

import torch
from torch.types import _size
from torch.distributions.beta import Beta

class ApproximatePosterior(Beta):
    """ ApproximatePosterior is the beta posterior ignoring the gamma parameter, and the sampling parameter"""
    def __init__(self, alpha, beta, gamma, Ns):
        super().__init__(alpha, beta)

## C-Beta distribution from the Continuous-Bernoulli paper
class ExactPosterior(torch.distributions.distribution.Distribution):
    """ Almost Exact Posterior --- we use weighted resampling or sampling importance resampling (SIR) """
    def __init__(self, alpha, beta, gamma, Ns):
        super().__init__()
        self.proposal = Beta(alpha, beta)
        self.gamma = gamma
        self.nsamples = Ns
        self.shape = torch.Size(alpha.shape)

    def __weights__(self, samples):
        w = ((torch.log(1-samples) - torch.log(samples))/(1-2*samples)); # ** self.gamma # using normalising factor of continous Bernoulli
        w = torch.cumsum(w, dim=(0))
        w = w / w[-1,:].unsqueeze(0)
        return w
        
    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        
        samples = self.proposal.sample(torch.Size([self.nsamples]) + sample_shape)
        w = self.__weights__(samples)
        u = torch.rand(sample_shape + self.shape).to(w.device)
        s = (w < u.unsqueeze(0))
        t = s.cumsum(dim=0)
        f, _ = torch.max(t, dim=0)
        z = torch.gather(samples, 0, f.unsqueeze(0)).squeeze(0)
        return z

def mean_prediction(post, Ns=100):
    """
    Get mean prediction via sampling
    """
    lbda = post.sample([Ns])
    # use linear between 0.4 and 0.6 to prevent problems with expression
    m = torch.where( torch.logical_and(lbda > 0.4, lbda < 0.6),
                     0.3370 * lbda + 0.3315,
                    lbda / (2*lbda-1) + 0.5/torch.atanh(1-2*lbda))
    m = torch.mean(m, dim=0)
    return m
