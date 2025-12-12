# Design Philosophy

## Data as a first class citizen

Stimulus is built to get the data processing right, which we argue is much harder and yields higher upsides than getting the model a bit more right (for getting the model right, we trust you!).

In this regard, scientific parameters (see google's blog [link to come]) are the data parameters. We designed the system to run ablation on the tools used to process the data, the kind of data selected in input etc. 

This means, you select data parameters, and a range of model parameters, stimulus will use Optuna to run hyperparameter optimization of your model on each of the data parameters and assume each model got to optimum performance thus efficiently comparing which dataset yields the best model.

## Minimal abstraction

When using stimulus, expect to write a lot of python code. It will be your dataset class, your model (with your training algorithm) and so fourth. 

The idea is that everything is fully transparent to you, you know exactly what is being ran, you can see the code, and you can debug it. We only abstract away the boilerplate code and the *boring stuff* like config parsing, logging, running replicates etc.

It will require a bit of extra effort, but this is the price to pay to understand what is actually going on. Do not worry though, we provide helpful placeholders and examples to get you started.



## Configurable and reproducible

In stimulus, we make extensive use of .yaml configs to define the training pipeline. This allows you to define the training pipeline in a human readable way, version control this file and share it. Since stimulus is built to be used in a containerized environment (best within it's parent nf-core pipeline, deepmodeloptim), it is then reproducible. 

## Works at scale

Stimulus is designed to be nextflow-friendly and devopped in tandem with it's parent nf-core pipeline, deepmodeloptim. This allows you to rebuild the stimulus container at any time with your changes and deploy it at scale in any HPC/AWS/GCP/Kubernetes environment within the deepmodeloptim framework.
