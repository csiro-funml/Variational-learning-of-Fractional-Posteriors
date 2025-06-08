"""
Encoder and Decoder for VAE experiments
Copied from the code by Ruthotto and Haber (2021) at https://github.com/EmoryMLIP/DeepGenerativeModelingIntro
- Modified for different sizes and channels
- EncoderSI which is then a modification of Encoder for semi-implicit distribution
- EncoderSIB is semi-implicit with coupled fractional and Bayes posterior
The experiments for MNIST and Fashion-MNIST uses d=2 number of convolutional layers.
The experiment for CIFAR10 (not in paper) use d=3. We probably need better neural network models to have a better absolute results for CIFAR10; but the current set up already gives relative results from which conclusions can be drawn.
"""

import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as Dist

class Generator(nn.Module):
    def __init__(self, s, c, d, w, q):
        """
        Initialize generator
        :param s: image size (multiples of 4)
        :param c: number of channels
        :param d: number of convolutional layers (2 or 3)
        :param w: number of channels on the finest level
        :param q: latent space dimension
        """
        super(Generator, self).__init__()
        scale = 2 ** d
        if s % scale !=0:
            raise Exception("Image size must be multiple of ", scale)
        b = s // scale; # basic image size
        
        self.b = b
        self.c = c
        self.d = d
        self.w = w
        if d == 2:
            self.fc = nn.Linear(q, w * 2 * b * b)           
        else:
            self.fc = nn.Linear(q, w * 4 * b * b)
            self.conv3 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(4*w)
        
        self.conv2 = nn.ConvTranspose2d(w * 2, w, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(w, c, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(w)
        self.bn2 = nn.BatchNorm2d(2*w)

    def forward(self, z):
        """
        :param z: latent space sample
        :return: g(z)
        """
        gz = self.fc(z)

        if self.d == 2:
            gz = gz.view(gz.size(0), self.w * 2, self.b, self.b)
        else:
            gz = gz.view(gz.size(0), self.w * 4, self.b, self.b)
            gz = self.bn3(gz)
            gz = F.relu(gz)
            gz = self.conv3(gz)
            
        gz = self.bn2(gz)
        gz = F.relu(gz)
        gz = self.conv2(gz)
        
        gz = self.bn1(gz)
        gz = F.relu(gz)
        gz = torch.sigmoid(self.conv1(gz))
        return gz

class Encoder(nn.Module):
    def __init__(self, s, c, d, w, q, naux=0):
        """
        Initialize the encoder for the VAE
        :param s: image size (multiples of 4)
        :param c: number of channels
        :param d: number of convolutional layers (2 or 3)
        :param w: number of channels on finest level
        :param q: latent space dimension
        """
        super(Encoder, self).__init__()
        scale = 2 ** d
        if s % scale !=0:
            raise Exception("Image size must be multiple of ", scale)
        b = s // scale; # basic image size
        
        self.b = b
        self.c = c
        self.d = d
        self.conv1 = nn.Conv2d(c, w, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(w, w * 2, kernel_size=4, stride=2, padding=1)

        if d==2:
            self.fc_mu = nn.Linear(w * 2 * b * b + naux, q)
            self.fc_logvar = nn.Linear(w * 2 * b * b + naux, q)
        else:
            self.conv3 = nn.Conv2d(w * 2, w * 4, kernel_size=4, stride=2, padding=1)
            self.fc_mu = nn.Linear(w * 4 * b * b + naux, q)
            self.fc_logvar = nn.Linear(w * 4 * b * b + naux, q)

    def forward(self, x_aux):
        """
        :param x_aux: MNIST image
        :return: mu,logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        x = x_aux[0] if isinstance(x_aux, tuple) else x_aux
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.d == 3:
            x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # handle implicits
        if isinstance(x_aux, tuple):
            aux = x_aux[1]
            if aux.dim() == 3:
                x = x.expand(aux.shape[0], -1, -1) # duplicate sample-wise                
            x = torch.cat((aux, x), dim=-1)
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# The implicit distribution of the Semi-implicit distribution
# See Yin and Zhou, ICML 2018.
class Implicit(nn.Module):
    """
    This class has an nMCMC configuration that controls how many implicit samples per encoded output
    If nMCMC=0, then it does one sample and collapse the sample output dimension (as if there is no MCMC)
    If nMCMC=1, then it does one sample, but does not collapse the sample output dimension (so that you know there is one sample)
    """
    def __init__(self, in_dim, eps_dim, hid_dim):
        """
        param in_dim: input dimensions
        param eps_dim: implicit/driving-noise dimensions
        param hid_dim: hidden layer dimensions
        """
        super(Implicit, self).__init__()

        self.eps_dim = eps_dim        
        self.fcs = nn.ModuleList([nn.Linear(eps_dim[0] + in_dim, hid_dim[0])]);
        for i in range(1, len(eps_dim)):
            self.fcs.append(nn.Linear(hid_dim[i-1] + eps_dim[i] + in_dim, hid_dim[i]));
        
    def forward(self, x):
        ## Yin and Zhou uses Salt and pepper and ReLU
        ## We use Normal because we are not using binarised images and
        ## We use LeakyReLU to prevent being stuck at zero, and the posterior may be degenerate
        ## We use Sigmoid at the last layer to visualise the implicits without worrying about scale

        if x.get_device() == -1:
            dist = Dist.Normal(torch.tensor([0.5]), torch.tensor([1.0]))
        else:
            dist = Dist.Normal(torch.tensor([0.5]).to(x.device), torch.tensor([1.0]).to(x.device))
        shape = list(x.shape)

        ## First layer does not have hidden units as inputs
        i=0;
        shape[-1] = self.eps_dim[i]
        eps = dist.sample(shape).squeeze(-1)
        h = F.leaky_relu(self.fcs[i](torch.cat((eps, x), dim=-1)))
        
        for i in range(1, len(self.eps_dim)-1):
            shape[-1] = self.eps_dim[i]
            eps = dist.sample(shape).squeeze(-1)
            h = F.leaky_relu(self.fcs[i](torch.cat((h, eps, x), dim=-1)))

        ## Last layer for plotting
        i=len(self.eps_dim)-1;
        shape[-1] = self.eps_dim[i]
        eps = dist.sample(shape).squeeze(-1)
        h = F.sigmoid(self.fcs[i](torch.cat((h, eps, x), dim=-1)))
            
        return h


# Encoder using semi-implicits for fractional posterior
class EncoderSI(nn.Module):
    def __init__(self, s, c, d, w, q, implicit_dim, implicit_hidden):
        """
        Initialize the encoder for the VAE
        :param s: image size (multiples of 4)
        :param c: number of channels
        :param d: number of convolutional layers (2 or 3)
        :param w: number of channels on finest level
        :param q: latent space dimension
        :param impliicit_dim, implicit_hidden: see class Implicit
        """
        super().__init__()

        Nfeat = c * s * s; # C * H * W
        
        # Set up the semi-implicit distribution
        self.nMCMC = 0;
        self.implicit = Implicit(Nfeat, implicit_dim, implicit_hidden) # the implicit part
        self.encoder = Encoder(s, c, d, w, q, implicit_hidden[-1])     # the explicit part

    @classmethod
    def make(cls, nMCMC, implicit, encoder):
        obj = cls.__new__(cls)  # Does not call __init__
        super(cls, obj).__init__()  # Don't forget to call any polymorphic base class initializers
        
        obj.nMCMC = nMCMC
        obj.implicit = copy.deepcopy(implicit)
        obj.encoder = copy.deepcopy(encoder)
        return obj
    
    def learn_implicit(self, requires_grad):
        for param in self.implicit.parameters():
            param.requires_grad = requires_grad

    def forward_implicit(self, x, Nsamples):
        x = x.view(x.size(0), -1) # flatten it
        if Nsamples != 0:
            x = x.expand(Nsamples, -1, -1)        
        return self.implicit(x)

    def forward_explicit(self, x, h):
        return self.encoder((x,h))
                
    def forward(self, x):
        """
        :param x: MNIST image
        :return: mu,logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        # For computational efficiency, we must not expand x to number of MCMC samples *before* the image stuff in encoder
        h = self.forward_implicit(x, self.nMCMC)
        qz = self.forward_explicit(x, h)
        return qz, h


# Encoder using semi-implicit for both Fractional posterior + Bayes posterior
class EncoderSIB(EncoderSI):
    def __init__(self, s, c, d, w, q, implicit_dim, implicit_hidden):
        """
        Initialize the encoder for the VAE
        :param s: image size (multiples of 4)
        :param c: number of channels
        :param d: number of convolutional layers (2 or 3)
        :param w: number of channels on finest level
        :param q: latent space dimension
        :param impliicit_dim, implicit_hidden: see class Implicit
        """
        super().__init__(s, c, d, w, q, implicit_dim, implicit_hidden)
        self.encoderb = Encoder(s, c, d, w, q, implicit_hidden[-1]) # extra Bayes posterior

    def fractional(self):
        """
        Get just the fractional poterior
        """
        return EncoderSI.make(self.nMCMC, self.implicit, self.encoder)
    
    def bayes(self):
        """
        Get just the bayes poterior
        """
        return EncoderSI.make(self.nMCMC, self.implicit, self.encoderb)

    def forward(self, x):
        """
        :param x: MNIST image
        :return: mu,logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        qz, h = super().forward(x)
        rz = self.encoderb((x,h))

        return qz, rz, h
