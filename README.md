# Mars-Spectrometry-Data-Driven

## Overview

This repo contains my code for the Mar Spectrometry challenge hosted by Data Driven. 

https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/

In this challenge, I built a machine learning model to automatically analyze mass spectrometry data collected for Mars exploration in order to help scientists in their analysis of understanding the past habitability of Mars. The model detects the presence of certain families of chemical compounds in data collected from performing evolved gas analysis (EGA) on a set of analog samples. 

See File Descriptions for details on running the code

The following image is the SAM Testbed on Mars and its replica at NASA that collected the data I processed

![image](https://github.com/RaviShah1/Mars-Spectrometry-Data-Driven/blob/main/plots/Target_Distrubutions.png)

The following chart shows the targets I am predicting and their distubutions

![image](https://github.com/RaviShah1/Mars-Spectrometry-Data-Driven/blob/main/plots/Target_Distrubutions.png)

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
- Group abundance for sample by m/z

**Mass Spectrometry Specific**
- Smooth Signal - applied savgol filter to smooth an abundance signal
- Baseline Subtraction - many methods attempted, best cross validation was produced by simply subtracting the minimum from each abundance signal
- Scale Abundance - scaled abundance from 0 to 1 across entire sample

**Feature Engineering**
- Binning - binned each m/z by temperature from range [-100,1600] with frequency of 100
- Find Peak - used max value (peak) at each bin for each m/z as feature

**Left Is Before Preprocessing, Right is After Smooth and Baseline Subtraction**

**Top is an example of a commerical sample, bottom is an example of a sam testbed sample**

![image](https://github.com/RaviShah1/Mars-Spectrometry-Data-Driven/blob/main/plots/Preprocess_Commercial_Example.png)

![image](https://github.com/RaviShah1/Mars-Spectrometry-Data-Driven/blob/main/plots/Preprocess_Sam_Testbed_Example.png)

## Machine Learning Modelling

**Model Used**

Light Gradient Boosted Machine (LGBM)
- metric = binary cross entropy
- reg alpha = 1
- cosample bytree = 0.4
- random state = 42

Hyperparam tuned to cross validation

**Failed Models**
- Logistic Regression
- KNN
- Simple MLP
- Guassian NB

**Cross Validation**
- For testing, I used stratisfied k-fold and train each target column separtely, see best results in chart below
- For training, I split and trained 10 folds using a multi label stratisfied k-fold

![image](https://github.com/RaviShah1/Mars-Spectrometry-Data-Driven/blob/main/plots/Cross_Validation_Chart.PNG)

## References
https://www.drivendata.co/blog/mars-spectrometry-benchmark/ 

https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/page/438/

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.398.2594&rep=rep1&type=pdf#:~:text=Preprocessing%20is%20the%20process%20that,this%20is%20an%20open%20problem.
