# Variational Learning of Fractional Posteriors

This repository accompanies the paper
<ins>Kian Ming A. Chai and Edwin V. Bonilla. Variational Learning of Fractional Posteriors. ICML 2025.</ins>

The experiments are in two places:

1. [Gaussian mixture models](notebooks/gmm.ipynb)
   - This is self-contained.
   - You can probably run it on your laptop. GPU is not required.
   - You can only use it within a Jupyter Notebook.

2. [Variational autoencoders for MNIST and Fashion-MNIST](notebooks/vae-expt.ipynb)
   - This depends on the [inference code](inf/vfbae.py) and the [neural network architectures](nnarch/vaenn.py).
   - You will need at least an NVIDIA T4 to run. Possible on free accounts on
     * [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/);
     * [Google Colab](https://colab.research.google.com/); and
     * [Kaggle](https:/kaggle.com).
   - Some results in the paper can be obtained within single time-limited sessions. Some results require multiple such sessions. Saving and loading of partial results are supported in the code. Multi-GPUs are supported via DataParallel, which is useful when run within Kaggle.
   - For FID scoring, we use the version by [Seitzer](https://github.com/mseitzer/pytorch-fid).
   - Instead of Jupyter Notebook, the code can be converted to pure Python and run from the command line. See the [experiments directory](expt/) for how this is done.
   - There is limited testing for CIFAR10, which requires some changes to execute on SageMaker Studio Lab.
     * One additional CNN layer (total 3) each for encoder and decoder
     * 32-dimensional latent space
     * Number of Monte Carlo samples for training reduced to 16
     * Number of Monte Carlo samples for validation reduced to 128
     * Numer of epoches reduced to 300
     * Results using $\mathcal{L}_\gamma$, our primary bound:
       - $\gamma=1$  (ELBO): train objecture = 1,229; validation objective = 1,172; FID = 141
       - $\gamma=10^{-5}$ (posterior very close to prior): train objective = 1,238; validation objective = 1,184; FID = 135
     * These results are not fantastic, but they demonstrate that small $\gamma$ is better.
     * The study is not extensive. In particular, the 32-dimensional latent space is just to show that the conclusions also hold beyond the 2 and 4 dimensions documented in the paper.
     