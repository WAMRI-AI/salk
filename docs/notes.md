
# Background

People:

- Uri Manor is the director of the imaging group at salk
- Linjing Fang is primary contact

# Setup

/scratch - data (not backed up)

* /scratch/bpho

- datasources

datasources is meant to be immutable named sources of data - they have a tendency to keep giving me random piles of images / czi and they tend to get mixed up so this is my attempt to fix that

- datasets

is a folder in bpho that is created dynamically by python files in the salk/uri/scirpts/dataset folder.
preprocessing is required before training - things like pulling images out of czi, normalizing the brightness etc.

- XXX?
there is a also a folder in bpho created dynamically that is models.
unlike fastai default i don't like saving models in my dataset folders - so i set the model path separately when creating my learners.

# Prerequisites

https://fiji.sc/ - microscopy people use it a lot to evaluate images and its good for opening czi files too

czi is a microscope format - there is a python module to open it czifile
It basically a multidimensional image array format bc microscopes can capture on many dimensions (x,y, z (depth), channel and time)

# Experiments

An experiment is typically one notebook - it uses a particular dataset and generates a single model. This way we can keep track of results - like if there is a result we like we know which dataset and notebok produced it. And we have code to recreate the dataset in the scripts/dataset folder One file per dataset

# Data Sources

- confocal_airyscan_pairs_mito
- live_neuron_mito_timelapse_for_deep_learning
- MitoTracker_Red_FM_movie_data
- Poisson-Gaussian_denoising_dataset

confocal_airyscan_pairs_mito - is probably what you should start with to get feet wet.
it is special in that it has paired low res and high res images.

live_neuron_mito_timelapse_for_deep_learning and MitoTracker_Red_FM_movie_data are both "movie" data sources

Poisson-Gaussian_denoising_dataset is something we aren't working on right now, it comes from another research paper and was produced outside salk

## Pairing low res and high res images.

This requires manual effort on the part of Linjing or people like her.
because it is hard to capture a perfrectly lined up image in both low and high. resolution - because you can't capture both simulataneously and these are living cells.

The process of taking a high and low res image and making them line up in x y coordinate space is called **registration**.

Right now our best result has come from training on this dataset using a unet style like jeremy does in his lesson.

## synthesize low res images from high res experiment

Another experiment we are in the middle of is figuring out - is there a way to synthesize low res images from high.

Because it is very easy to capture high res and avoids pain of registration.

so we would have a lot more data to work with

This is where we are in the process right now is figuring that out - there are 3 possible approaches, 1 I have tried and it didn't work so well but that may just be i haven't tried hard enough.

## Getting started

My thought was to start - get you training the paired unet model to get familiar and also your card has less vram so need to make sure we can work with that

Once thats done we could move onto the part where we don't have answers yet - maybe thats when we talk in person more

## Notebook

1) salk/uri/scripts/dataset/paired_001.py will create the /scratch/bpho/dataset/paired_001 directory

it basically takes the paired images and creates matching ROI (region of interest) tiles of different sizes that we use for training

2) the notebook you want is salk/uri/paired_001_unet.ipynb

The top half of it trains models, at the bottom i have out of sample code that converts some movie files from LR to HR, you can probably figure that part out but i wouldn't worry about it til after we talk later.

For now just see if you can find batch sizes that work for you w/ the various tile sizes.

## Information Protection

Use the private github salk repo for notes and can change things around.

Until the paper is out - no public discussion of what we are working on or relationship w/ salk

The image data is private

I think they may release some publicly when we are done, the source code will become public for sure
