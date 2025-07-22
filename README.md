# Importance Weighted Autoencoders
This repository contains code and resources for experiments conducted to explore Importance Weighted Autoencoders (IWAE) and their advantages over standard Variational Autoencoders (VAE). The experiments are implemented in Python using PyTorch and follow the methodology from Burda et al. (2016).
## Experiments Overview
The experiments are designed to investigate specific aspects of the IWAE framework and validate theoretical claims:

**Experiment 1: Active Latent Dimensions and Weight Initialization Methods**
* Objective: Investigate whether different weight initialization methods (Xavier vs. He, Uniform vs. Normal) lead to different total numbers of active latent dimensions in VAE and IWAE models. 
**Experiment 2: Sensitivity of Active Latent Dimensions Across Random Seeds**
*Objective: Analyze the variability of specific active latent dimensions across different random seeds to understand whether the selection of active dimensions is deterministic or stochastic. 

## Repository Contents

iwae.py: Core implementation containing VAE and IWAE model architectures, training procedures, and loss computations. Includes importance sampling mechanisms and the k-sample importance weighting framework (forked from borea17).
Other files: 
* Visualization utilities for generating plots of training losses, active latent dimensions heatmaps, and model comparison charts.
* Data processing utilities for computing active latent dimension statistics, aggregating results across multiple seeds, and calculating evaluation metrics.
* TensorBoard logging utilities for monitoring training progress, loss curves, and model performance metrics during experiments.

## Key Findings

IWAE consistently learns more active latent dimensions than VAE when using multiple importance samples (k=5), confirming superior latent space utilization
Weight initialization methods (Xavier vs. He) show minimal impact on active latent dimension counts, with differences falling within experimental variance
While the total number of active latent dimensions remains stable across seeds (~20.2 ± 0.2), individual dimension selection varies significantly, suggesting stochastic training and optimization effects.

## Experimental Setup

Dataset: Binarized MNIST
Architecture: Two-layer encoder/decoder with tanh activation
Latent dimensions: 50
Importance samples: k ∈ [1, 5]
Weight initializations: Xavier Normal/Uniform, He Normal/Uniform
Training epochs: 80
Random seeds: [135, 630, 924, 10, 32]

## References
For the theoretical foundation and original IWAE framework, please refer to:
Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). Importance Weighted Autoencoders. International Conference on Learning Representations (ICLR).
This work builds upon and extends the experimental validation of the importance weighting mechanism in deep latent variable models.