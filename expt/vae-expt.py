#!/usr/bin/env python
# coding: utf-8

# # Variational Autoencoder for Image Generation
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/VAE.ipynb)
# 
# ## Some Reference
# 
# - Original paper by [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114) and [Rezende et al. (2014)](https://arxiv.org/abs/1401.4082)
# - Comprehensive review by [Kingma and Welling (2019)](https://arxiv.org/abs/1906.02691)
# - Introduction by [Ruthotto and Haber (2021)](https://arxiv.org/abs/2103.05180)
#   - Our code is based on [theirs](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro)
# - Kian Ming A. Chai (Data61)'s Variational Learning of Fractional Posteriors (unpublished)

# ## Setup
# ### Key parameters

# In[1]:


### Data
batch_size = 64 # Used by torch.utils.data.DataLoader for mini-batching for training

### Model parameters
gamma=0.5                 # Parameter for fractional posterior we run with 1, 0.9, 0.5 and 0.1 **commandline configurable**
theoryclass = "VFBAESI"   # VFAE, VFAESI, VFBAESI **commandline configurable**

#### Encoder and Decoder of VAE
q = 2 # dimension of latent variabl $z$, so that we can plot and see it, from (Ruthotto and Haber, 2021)
width_enc = 32 # from (Ruthotto and Haber, 2021)
width_dec = 32 # from (Ruthotto and Haber, 2021)

##### Semi-implicit distribution#, see Yin and Zhou, ICML 2018.
implicit_dim = [15, 10, 5]    # implicit noise dimensions. We make it smaller than because we want ...
implicit_hidden = [28, 14, 2] # the implicit hidden output dimension to be eventually 2 dimensions, so that we can plot and see it

### Inference parameters
#### - The first number of is the number of samples for which data points are generated
#### - For stochastic encoder, the second number is the number of samples from the implicit distribution, generating the first.
#### - For Bayes+Fractional Posterior, the third number is the number of subsamples from the implicit distribution for cross evaluations
nMCMCtrain= (100, 10, 5) # For training
nMCMCval = (1024, 32, 16)  # For testing/validation

### Training parameters
num_epochs = 10             # number of training epochs **commandline configurable**
retrain = False              # **commandline configurable**
learning_rate = 1e-3         # input to Adam optimiser
learning_weight_decay = 1e-5 # input to Adam optimiser

### These are not so key
gpudevice = "cuda:3"   # **commandline configurable**
dovalidate = True     # **commandline configurable**


# ## System setup
# - to run from command line, convert this notebook with jupyter nbconvert filename.ipynb --to python
# - to run from Google colab, download and copy to your drive

# In[2]:


### Check Colab
import sys,os

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir('/content/drive/MyDrive/')
    dgm_dir = '/content/drive/MyDrive/vlfp/'
    if not os.path.exists(dgm_dir):
        raise Exception("Code does not exists. Download and copy to your drive under the directory vlfp")
    else:
        print("Code already exists")
    sys.path.append(os.path.dirname(dgm_dir)) # Allow our own modules
    os.chdir('/content/drive/MyDrive/vlfp/notebooks/')

    # sensible settings within (free) collab to override key_parameters
    is_interactive = lambda: True
    batch_size = 2048 if batch_size > 2048 else batch_size
    num_epochs = 20 if num_epochs > 20 else num_epochs
    gpudevice = "cuda:0"
else:
    import __main__ as main
    sys.path.append(os.path.abspath('..')) # Allow our own modules
    is_interactive = lambda: not hasattr(main, '__file__')


# In[3]:


### Nice plotting
import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'image.interpolation' : 'none'})
#plt.rcParams['figure.figsize'] = [10, 6]
#plt.rcParams['figure.dpi'] = 200


# In[4]:


### Perhaps Load the parameters from commandline
if not is_interactive():
    # we have a chance to change the fixed parameters
    print("Using commandline arguments: %s" % (sys.argv));

    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("--gamma", type=float)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--theoryclass", type=str, choices=("VFAE", "VFAESI", "VFBAESI"))
    argparser.add_argument("--device", type=str)
    argparser.add_argument("--skiptrain", action="store_true")
    argparser.add_argument("--skipvalidate", action="store_true")
    args = argparser.parse_args()

    gamma = args.gamma
    num_epochs = args.epochs
    theoryclass = args.theoryclass
    gpudevice = args.device
    retrain = False if args.skiptrain else True # this would be the default
    dovalidate = False if args.skipvalidate else True # this would be the default


# In[5]:


out_file = ("./results/%s-mnist-q-%d-e-%d-d-%d-g-%0.2f-n-%d-b-%d") % (theoryclass, q, width_enc, width_dec, gamma, nMCMCtrain[0], batch_size)


# In[6]:


### print Key parameters for logging
print("theoryclass=%s, batch=%d, gamma=%0.2f, q=%d, enc=%d, dec=%d, epochs=%d, MCMCtrain=(%d %d %d)" % ((theoryclass, batch_size, gamma, q, width_enc, width_dec, num_epochs) + nMCMCtrain))
print("File=%s" % (out_file))


# ## Prepare Image Data
# 
# https://pytorch.org/vision/stable/transforms.html#supported-input-types-and-conventions
# - Tensor image are expected to be of shape (C, H, W), where C is the number of channels, and H and W refer to height and width. Most transforms support batched tensor input. A batch of Tensor images is a tensor of shape (N, C, H, W), where N is a number of images in the batch. The v2 transforms generally accept an arbitrary number of leading dimensions (..., C, H, W) and can handle batched images or batched videos.
# 
# For data sets, see https://pytorch.org/vision/stable/datasets.html

# In[7]:


import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

img_transform = torchvision.transforms.ToTensor()

train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)

print(train_dataset, '\n\n', test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

x,_ = next(iter(train_dataloader))
fig = plt.Figure()
if hasattr(fig, "show"):
   plt.imshow(torchvision.utils.make_grid(x,16,padding=1,pad_value=1.0)[0], vmin=0, vmax=1, cmap='Greys')
   plt.axis("off")
   plt.margins(0, 0)
   plt.show()


# ## Setup Model

# In[8]:


import torch
from nnarch.vaenn import *
from inf.vfbae import *

device = torch.device(gpudevice if torch.cuda.is_available() else "cpu")

g = Generator(width_dec,q)

if theoryclass == "VFAE":
    e = Encoder(width_enc,q)
    VAE = VFAE
    # Just need one number
    nMCMCtrain = nMCMCtrain[0]
    nMCMCval = nMCMCval[0]
elif theoryclass == "VFAESI":
    e = EncoderSI(width_enc, q, implicit_dim, implicit_hidden)
    VAE = VFAESI
elif theoryclass == "VFBAESI":
    e = EncoderSIB(width_enc, q, implicit_dim, implicit_hidden)
    VAE = VFBAESI
else:
    raise ValueError("Unknown theory class " + theoryclass)

vae = VAE(e, g, gamma).to(device)


# ## Train the Generator and Approximate Posterior

# In[9]:


def singleEpoch(model, dataloader, nMCMC, optimizer = None):
    num_ex = 0

    stat = None

    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        scores = model.ELBOscores(image_batch, nMCMC)

        if optimizer is not None:
            optimizer.zero_grad()
            scores[0].backward()
            optimizer.step()
            
        this_num_ex = image_batch.shape[0]
        this_scores = np.array([ scores[0].item(), *scores[1:]]);

        num_ex += this_num_ex;
        if stat is None:
            stat = this_scores * this_num_ex
        else:
            stat += this_scores * this_num_ex

    stat /= num_ex
    return stat
    
def stat2str(t):
    return tuple(f"{num:0.3f}" for num in t);


# ### Here, we use ADAM, a stochastic approximation scheme that operates on minibatches.

# In[10]:


from os import path
import numpy as np

if retrain == False and path.exists(out_file + ".pt") and path.exists(out_file + ".mat"):
    vae.load_state_dict(torch.load("%s.pt" % (out_file),map_location=torch.device(device)))
    from scipy.io import loadmat
    his_file = loadmat("%s.mat" % (out_file))
    his = his_file['his']
    print("loaded results from %s" % out_file)
else:
    if out_file is not None:
        import os
        out_dir, fname = os.path.split(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print((3*"--" + "out_file: %s" + 3*"--") % (out_file))
    
    print((3*"--" + "device=%s, q=%d, batch_size=%d, num_epochs=%d, w_enc=%d, w_dec=%d" + 3*"--") % (device, q, batch_size, num_epochs, width_enc, width_dec))
        
    his = []
    vae.train()
    print("Total number of parameters = ", sum(p.numel() for p in vae.parameters()))
    #if hasattr(vae.e, "learn_implicit"): 
    #    vae.e.learn_implicit(False) # turn off first
    #    print("Learning %d parameters" % (sum(p.numel() for p in vae.parameters() if p.requires_grad)))
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=learning_weight_decay)
    for epoch in range(num_epochs):
        trainstats = singleEpoch(vae, train_dataloader, nMCMCtrain, optimizer)        
        print(("%06d  %s  ") % (epoch + 1, stat2str(trainstats)))
        his = his + [trainstats]
        # if hasattr(vae.e, "learn_implicit"):
        #    if epoch == (num_epochs // 5):
        #        vae.e.learn_implicit(True) # now we learn
        #        print("Learning %d parameters" % (sum(p.numel() for p in vae.parameters() if p.requires_grad)))
        #        optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=learning_weight_decay)
    
    his = np.stack(his)

    if out_file is not None:
        torch.save(vae.g.state_dict(), ("%s-g.pt") % (out_file))
        torch.save(vae.state_dict(), ("%s.pt") % (out_file))
        from scipy.io import savemat
        savemat(("%s.mat") % (out_file), {"his":his})


# ## Testing and Evaluation

# In[11]:


vae.eval();


# ### Print the fit with more MCMC samples and also ELBO
# - TODO: MCMC Error bars. Complicated because MCMC averaging done within

# In[12]:


def printtable(desc, scores):
    s = tuple(f"{num:0.3f}" for num in scores)
    print(desc, s)
    return " & ".join(s)


# In[13]:


if dovalidate:
    print("Setting: theoryclass=%s gamma=%.2f num_epochs=%d" % (theoryclass, gamma, num_epochs));
    s = theoryclass + " & " + f"{gamma:0.1f}"
    with torch.no_grad():        
        s = s + " & " + printtable("Train: ", -singleEpoch(vae, train_dataloader, nMCMCval));
        s = s + " & " + printtable("Validation: ", -singleEpoch(vae, test_dataloader, nMCMCval));
        if theoryclass == "VFBAESI":
            other = VFAESI(vae.e.bayes(), vae.g, 1.0)
            scores = -singleEpoch(other, test_dataloader, nMCMCval)
        else:
            vae.gamma = 1.0 # We do ELBO here
            scores = -singleEpoch(vae, test_dataloader, nMCMCval)
            vae.gamma = gamma # Make sure we are the correct setting
        s = s + " & " + printtable("Validation ELBO: ", scores) + " \\\\"
    print(s)


# ### Check implicit distribution (only for semi-implicit latents)

# In[14]:


if not hasattr(vae, "ELBO_and_implicit"):
    print("No ELBO_and_implicit")
else:
    from torch.utils.data import SequentialSampler
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SequentialSampler(train_dataset))    
    vae.e.nMCMC = 0 # to do for the rest for evaluation!

    nsamples = 500  #get 500 implicit samples from the first batch (64 images)
    implicits = []
    with torch.no_grad():
        image_batch,_ = next(iter(dataloader)) 
        image_batch = image_batch.to(device)
        implicit = vae.e.forward_implicit(image_batch, nsamples)
        implicits.append(implicit)
    implicits = torch.cat(implicits,0).cpu().numpy()

    fig=plt.Figure()
    if hasattr(fig, "show"):
        fig, axs = plt.subplots(8, 8)
        for i in range(8):
            for j in range(8):
                toplot = implicits[:, i*8+j, :]
                axs[i,j].axis((-0.01, 1.01, -0.01, 1.01))
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                axs[i,j].scatter(toplot[:,0],toplot[:,1], s=1)
        plt.show()
        
    # subsample to 25 images for saving
    implicits = implicits[:, 0:25, :].transpose(1,0,2).reshape(-1, 2)
    indx = np.tile(range(25), (nsamples, 1)).T.reshape(-1,1);
    implicits = np.hstack((indx, implicits))
    np.savetxt("%s-is.csv" % out_file , implicits, fmt="%d,%f,%f")

    # check for all image, one sample per image
    implicits = []
    with torch.no_grad():
        for image_batch, _ in test_dataloader:
            image_batch = image_batch.to(device)
            implicit = vae.e.forward_implicit(image_batch, 0)
            implicits.append(implicit)
    implicits = torch.cat(implicits,0).cpu().numpy()
    
    fig=plt.Figure()
    if hasattr(fig, "show"):
        plt.scatter(implicits[:,0], implicits[:,1])
    plt.show()
    
    # subsample to have 5,000 rows only, since we can only plot 5,000 in TikZ anyway
    np.savetxt("%s-ia.csv" % out_file , implicits[::2], fmt="%f,%f")

    id=47 # This is for semi-implicit ELBO, gamma=1.0
    with torch.no_grad():
        image_batch, label = train_dataloader.dataset[id]
        image_batch = image_batch.to(device).unsqueeze(0)
        
        us = []; elbos = []
        for i in range(0,500):
            nelbo, u = vae.ELBO_and_implicit(image_batch, (1000, 1, 500))
            nelbo = nelbo.cpu()
            u = u.squeeze(0,1).cpu().numpy()
            us.append(u); elbos.append(-nelbo)
        us = np.array(us); elbos = np.array(elbos)
    
        fig=plt.Figure()
        toplot = implicit.squeeze(1)
        plt.title("%d" % (id))
        plt.axis((-0.01, 1.01, -0.01, 1.01))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(us[:,0],us[:,1], c=elbos, cmap="Greys")
        plt.show()


# ### Show Samples
# - Test samples
# - 2-D interpolation in latent space

# In[15]:


ncol = 9
x0, _ = next(iter(test_dataloader))
x1, _ = next(iter(test_dataloader))
x = torch.concatenate( (x0, x1), dim=0 )[0:(ncol*ncol), :, :, :]

x_samples = torchvision.utils.make_grid(x.cpu().detach(),ncol,1,pad_value=1)[0]
plt.imsave("%s-testset-9x9.png" % out_file, x_samples.cpu().numpy(), vmin=0, vmax=1, cmap='Greys');

fig = plt.Figure(frameon=False)
if hasattr(fig, "show"):
    plt.subplot(1,2,1)
    plt.imshow(x_samples, vmin=0, vmax=1, cmap='Greys')
    plt.axis("off")
    plt.margins(0, 0)

from statistics import NormalDist
with torch.no_grad():
    pam = torch.from_numpy(np.array([NormalDist().inv_cdf(x) for x in np.linspace(0.1, 0.9, ncol)])).float().to(device)
    mxx, myy = torch.meshgrid(pam, pam)
    mxx = mxx.flatten().unsqueeze(1)
    myy = myy.flatten().unsqueeze(1)
    mzz = torch.cat( (mxx, myy), dim=1)
    x_vae = vae.mean_x(mzz) #  Since we are only testing the decoder, the encoder (whichever posterior) doesn't matter

x_samples = torchvision.utils.make_grid(x_vae.cpu().detach(),ncol,1,pad_value=1)[0]
plt.imsave("%s-2d.png" % out_file, x_samples.cpu().numpy(), vmin=0, vmax=1, cmap='Greys');

if hasattr(fig, "show"):
    plt.subplot(1,2,2)
    plt.imshow(x_samples, vmin=0, vmax=1, cmap='Greys')
    plt.axis("off")
    plt.margins(0,0)
    plt.show()


# ### Prepare for subsequent steps

# In[16]:


mu_val = []
label_val = []
loglike_val = []
logprior_val = []
zs_val = []
vae.e.nMCMC=0
for x, label_batch in test_dataloader:
        with torch.no_grad():
            x = x.to(device)
            s = x.shape[1]
            mu, logvar = vae.encode(x)
            z, _ = vae.sample_ezx(mu, logvar)
            gz = vae.g(z)

            log_pzx, log_like, log_prior = vae.log_prob_pzx(z, x, gz)

            zsamples, _ = vae.sample_ezx(mu, logvar, 10) # this is missing from the Tutorial's code
            zsamples = zsamples.view( (-1, zsamples.shape[-1]))

            mu_val.append(mu)
            label_val.append(label_batch)
            loglike_val.append(log_like)
            logprior_val.append(log_prior)

            zs_val.append(zsamples);

mu_val = torch.cat(mu_val,0).cpu().numpy()
label_val = torch.cat(label_val,0).cpu().numpy()
loglike_val = torch.cat(loglike_val,0).cpu().numpy()
logprior_val = torch.cat(logprior_val,0).cpu().numpy()
zs_val = torch.cat(zs_val,0).cpu().numpy()


# ### Show Posterior Approximation

# In[17]:


## 5) show joint probability p(z,x) and approximation e(z|x) for some examples
imin = np.argsort(loglike_val)[:1]
imax = np.argsort(loglike_val)[-1:]

with torch.no_grad():
    vae.eval()
    z1 = torch.linspace(-5.5, 5.5, 100)
    z2 = torch.linspace(-5.5, 5.5, 100)
    zg = torch.meshgrid(z1, z2)
    z = torch.cat((zg[0].reshape(-1, 1), zg[1].reshape(-1, 1)), 1).to(device)

    # reconstruct images from the latent vectors
    gz = vae.g(z)

    # compute posterior
cnt=0
for ind in np.hstack((imin,imax)):
    vae.eval()
    print("ind=%d" % ind)
    x, _ = test_dataloader.dataset[ind]
    x = x.to(device).unsqueeze(0)
    zt = vae.encode(x)[0].detach()
    gzt = vae.g(zt)
    log_pzx, log_like, log_prior = vae.log_prob_pzx(z, torch.cat(len(z1)*len(z2)*[x]), gz)
    log_ezx = vae.log_prob_ezx(z,torch.cat(len(z1)*len(z2)*[x]))
    zm = z[np.argmax(log_pzx.cpu().numpy()), :]
    gzm = vae.g(zm.unsqueeze(0))

    fig=plt.Figure()
    if hasattr(fig, "show"):
        plt.subplot(cnt+1,5,1+cnt*5)
        img = log_pzx.reshape(len(z1),len(z2))
        plt.contour(z1,z2,img.cpu().numpy(),40,linewidths=2)
        plt.plot(zm[1].cpu(),zm[0].cpu(),"bs")
        plt.plot(zt[0,1].cpu(),zt[0,0].cpu(),"or")
        plt.xlabel(r"$\mathbf{z}_1$", labelpad=-20)
        plt.ylabel(r"$\mathbf{z}_2$", labelpad=-30)
        plt.axis('square')
        plt.axis((z1[0], z1[-1], z2[0], z2[-1]))
        plt.xticks((z1[0], z1[-1]))
        plt.yticks((z2[0], z2[-1]))
        if cnt==0:
            plt.title("log_pzx")

        plt.subplot(cnt+1,5,2+cnt*5)
        img = log_ezx.reshape(100,100)
        plt.contour(z1,z2,img.detach().cpu().numpy(),30,linewidths=2)
        plt.plot(zt[0,1].cpu(),zt[0,0].cpu(),"or")
        plt.xlabel(r"$\mathbf{z}_1$", labelpad=-20)
        plt.ylabel(r"$\mathbf{z}_2$", labelpad=-30)
        plt.axis('square')
        plt.axis((z1[0], z1[-1], z2[0], z2[-1]))
        plt.xticks((z1[0], z1[-1]))
        plt.yticks((z2[0], z2[-1]))
        if cnt==0:
            plt.title("log_ezx")

        plt.subplot(cnt+1,5,3+cnt*5)
        plt.imshow(x.cpu().squeeze(0).squeeze(0), vmin=0, vmax=1, cmap='Greys')
        # plt.axis('square')
        plt.axis('off')
        plt.margins(0, 0)
        if cnt==0:
            plt.title("true image")

        plt.subplot(cnt+1,5,4+cnt*5)
        plt.imshow(gzt[0].detach().cpu().squeeze(0).squeeze(0), vmin=0, vmax=1, cmap='Greys')
        # plt.axis('square')
        plt.axis('off')
        plt.margins(0, 0)
        if cnt==0:
            plt.title("MAP e_zx")

        plt.subplot(cnt+1,5,5+cnt*5)
        plt.imshow(gzm[0].detach().cpu().squeeze(0).squeeze(0), vmin=0, vmax=1, cmap='Greys')
        plt.axis('off')
        plt.margins(0, 0)
        if cnt==0:
            plt.title("MAP p_zx")
    cnt+=1


# ### Show Latent Space Structure

# In[ ]:


fig=plt.Figure()
if hasattr(fig, "show"):
    plt.subplot(1,2,1)
    plt.scatter(mu_val[:,0],mu_val[:,1],c=label_val)
    #plt.axis("square")
    #plt.axis((-3.5, 3.5, -3.5, 3.5))
    plt.xticks((-3.5, 3.5))
    plt.yticks((-3.5, 3.5))
    plt.xlabel(r"$\mathbf{z}_1$", labelpad=-20)
    plt.ylabel(r"$\mathbf{z}_2$", labelpad=-30)

    plt.subplot(1,2,2)
    plt.hist2d(zs_val[:,0],zs_val[:,1],bins=256,range=[[-3.5,3.5],[-3.5,3.5]])
    plt.axis("square")
    plt.axis((-3.5, 3.5, -3.5, 3.5))
    plt.xticks((-3.5, 3.5))
    plt.yticks((-3.5, 3.5))
    plt.xlabel(r"$\mathbf{z}_1$", labelpad=-20)
    plt.ylabel(r"$\mathbf{z}_2$", labelpad=-30)


# In[ ]:


# save for plotting in LaTeX
# subsample to have 5,000 rows only, since we can only plot 5,000 in TikZ anyway
np.savetxt("%s-mu.csv" % out_file , (np.concatenate((mu_val, np.expand_dims(label_val,1)), axis=1))[::2], fmt="%f,%f,%d")
np.savetxt("%s-zs.csv" % out_file , zs_val[::20], fmt="%f,%f")


# In[ ]:




