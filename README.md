# VoiceMed's Transfer Learning

This repository contains a baseline to build Transfer Learning models of any type,
based on [yamnet](https://tfhub.dev/google/yamnet/1).

## CoughDetection

In this repository the model is used to improve our cough detector, the performance boost is
major and can be observed in the W&B study [here](https://wandb.ai/mastersplinter/EE).

The way this implementation works is by importing the model from TFHub, and using Yamnet as
a feature extractor. This is done in `process.py`, each audio sample is loaded and parsed to the
model. The model then processes the signal and returns predictions (a 512 vector since the original model has 512 classes),
embeddings and log_mel_spectrogram. The embeddings are the output of the Convolutional layers of the
model and this is what will be used. We save all the embeddings and use it as our training data.

The custom model made is created with 2 Dense layers (`model.py`) which take the embeddings as input (shape of 1024, 1).
This allows really fast feature extraction because it's done by the convolutions of Yamnet, and really
fast training as the model complexity is really low.

## Files

- `Yamnet/` (contains code of yamnet, it's not used or needed but it's good reference)
- `model.py` (contains last layers of the network)
- `process.py` (contains processing using yamnet)
- `yamnet.h5` (weights for yamnet, not needed right now but good for possible expansion)

## Flexibility

This allows us to use this technique to quickly build any kind of model we want (like the biomarkers for MIT testing),
because the processing in automated and the only thing that needs to be changed is the few Dense layers.

## Expansion

I have tried for a while to try and import the whole Yamnet and re-train it on our own dataset (instead of only last Dense layers),
however I have not succeeded for some data shape errors I have encountered. I will continue researching about this
because it would provide a more robust model where even the pre-processing automatically improves for the given implementation.
