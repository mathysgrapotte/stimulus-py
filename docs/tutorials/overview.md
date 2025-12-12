# Overview

Stimulus defines a some key concepts

## StimulusDataset

This is the main interface between your data and stimulus. It explains stimulus how to load it from disk, how to split it, convert it to a PyTorch dataset etc.

## StimulusModel 

The placeholder for your model, it is a simple PyTorch interface with two extra required methods, one for a single training step and one for evaluating on a batch of data. 

The `validation` method is called through the entire framework for metrics and late stage model validation on the held out test set. 

Your StimulusModel lives in an external `.py` file so that you can version-control it and share it arround. 

## Split

The act of splitting a dataset, this is done through applying a `splitter` class to a StimulusDataset object. 

The `splitter` class simply takes a StimulusDataset object and returns two list of indices, one for the train split, one for the eval split. 

With those two lists, the `StimulusDataset` object will split itself in two. 

You wonder, where is the test split ? This process is intentionally not automated, we believe your test split should be carefully engineered, far away from your training data, so you will pass this as an argument during evaluation. 

## Transform 

The act of transforming a dataset, we define transformations as in-place operations on the dataset. Those might add or remove fields (columns). Stimulus is designed to chain as many transformations together as we want. 

Transformations are defined in a `transform` class that can apply either to the entire dataset or to only a few elements, in which case `StimulusDataset` will paralelize the operation seemlessly. 

## Encode 

The act of encoding a dataset, this is the last step before training and is responsible for converting human readable data into machine readable data. Example of encoders are tokenizers, one-hot encoders etc. 

Since those often play an important role in model's performance, we decided to seperate them from transformations but made the encode step optional. 

## Data config

The data config determines which operations to apply on the data (how to split, transform, encode etc.), Which global seed to use, etc. 

This config is thought to be central and is later split into subconfig for parallelization (so subconfig will contain only a single transformation while the master config will contain all transformations). 

This is to ensure traceability accross the entire framework. 

## Model config 

The model config defines various parameters that can be swept over during tuning, default parameters (unswept parameters) should be set to a default value in the model's `__init__` method. 
