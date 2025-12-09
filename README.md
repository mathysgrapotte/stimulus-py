# Stochastic Input Modulation for Unbiased Learning Systems (stimulus)

[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mathysgrapotte.github.io/stimulus-template/)
[![Build with us on slack!](http://img.shields.io/badge/slack-nf--core%20%23deepmodeloptim-4A154B?labelColor=000000&logo=slack)](https://nfcore.slack.com/channels/deepmodeloptim)

Many Deep Learning (DL) models in biology report strong premises but fail in practice. 
We see hidden leakage, shortcut learning and distribution shift in almost every attempt at using bioDL models in production environments.

Common Machine Learning Operation (MLops) frameworks are not designed for bioDL usecases, mostly because they can't handle biological datasets ! 

Over the years, in our field, the community developped a wide array of tools to seperate signal from noise. Those tools live in different programming languages, different communities, different containers. The range of those tools far exceeds what is available in MLops framework for conventional DL applications (language, vision etc.).

Consequently, bioDL models are trained in isolation from the data-processing choices, which we believe explains most of the issues we see in the field.

STIMULUS is a MLops framework taking the best of both worlds, modern DL engineering practices and bio specific tooling, resulting in a framework specifically designed for training production-ready bioDL models. 

## Principles

### Data as a first class citizen

We believe that deep learning as a field is vastly explored. Training pipelines, custom kernels, hyperparameter choices, model architectures, etc. are all well understood compared to how to process biological data. Therefore, we provide a wide array of tools to tune and systematically explore the way biological data is processed and focus on this rather than model tuning. 

I.e. model tuning is constant accross all experiments (nuisance parameter), data processing is the scientific variable.

Since different datasets require different model hyperparameter choices, stimulus provides built-in hyperparameter solution with Optuna. 

### Minimal abstraction

We follow a principle of minimal abstraction, maximal flexibility. The idea is to split processes into two categories, mutable and immutable. 

We define immutable processes by those that do not change scientific outcome. One example would be, dispatching a job to a slurm grid, or converting a .h5ad file into a parquet file. 

We define mutable processes by those that do change the scientific outcome. One example would be, if I use this model architecture or that model architecture. 

We choose to only abstract immutable processes, so, while using stimulus, you will see yourself writing quite a bit of python code (your own models, how your dataset is processed etc.). We designed the framework so that this process is easy, through naming conventions, good project structure etc. 

### Configurable and reproducible

In stimulus, we make extensive use of .yaml configs to define the training pipeline. This allows you to define the training pipeline in a human readable way, version control this file and share it. Since stimulus is built to be used in a containerized environment (best within it's parent nf-core pipeline, deepmodeloptim), it is then reproducible. 

### Works at scale

Stimulus is designed to be nextflow-friendly and devopped in tandem with it's parent nf-core pipeline, deepmodeloptim. This allows you to rebuild the stimulus container at any time with your changes and deploy it at scale in any HPC/AWS/GCP/Kubernetes environment within the deepmodeloptim framework.



