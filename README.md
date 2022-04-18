# Mars-Spectrometry-Data-Driven

## Overview

## File Descriptions
**Folders**
- data - data used for training and inference
- plots - files and images created for EDA
- saved_models - pkl files containing trained models
- submissions - csv files containing competition submissions

**Files in Main**
- run.sh - example of file to run to generate a dataset folder, train a model, and create a submission (runs full pipeline)
- generate_dataset.py - preprocesses and feature engineers data (uses argparse for command line use)
- train_pipeline.py - runs training code (uses argparse for command line use)
- inference_pipeline.py - run inference code (uses argparse for command line use)
- preprocess.py - contains preprocessing functions to structure data, apply signal smoothing, remove baseline, and normalize signal
- feature_engineering.py - contains function to get features for machine learning model (bins signal and gets peak for each bin)
- models.py - contains machine learning models to train for task
- cross_validation.py - used to get cross validation scores while testing models
- requirements.txt - pip install to recreate python environment used to train models

## Preprocessing
**Unstructured -> Structured**
- Drop m/z values above 100 since all samples had m/z in range [0,99]
- Dop m/z 4 (Helium) since that was the carrier gas
- 

## Machine Learning Modelling

## References
https://www.drivendata.co/blog/mars-spectrometry-benchmark/ 

https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/page/438/

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.398.2594&rep=rep1&type=pdf#:~:text=Preprocessing%20is%20the%20process%20that,this%20is%20an%20open%20problem.
