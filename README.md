# TRISAR: Self-Supervised Triplet Metric Learning for Temporal SAR Interpretation

This repository was created for the 2026 IEEE Data Fusion Contest.

The project has 3 main parts:

1. Download data
2. Train the model
3. Run the demo

## Installation

Install the required packages with:

pip install -r requirements.txt

## Project structure

The main parts of the project are:

- download utilities
- training utilities
- demo application

## 1. Download data

The first step is to filter and download the required SAR TIFF files.

This part is used to:
- select scenes near predefined locations
- download the TIFF files
- organize them into the data folders
- prepare patch datasets and CSV files

## 2. Train

The training part is used to:
- load the generated triplets and image data
- train the TRISAR model
- save checkpoints and outputs

## 3. Demo

The demo is a Streamlit application.

It can be used to:
- inspect downloaded TIFF scenes
- analyze patch timelines
- visualize temporary and continuous changes
- localize the most suspicious regions

## Run the demo

Start the demo with:

streamlit run demo/trisar_app.py

You can also pass parameters from the command line.

## Notes

- Large TIFF files and generated datasets should not be uploaded to Git
- Model checkpoints should also be ignored unless you explicitly want to share them
- The repository is mainly for code, configuration, and notebooks

## Summary

This repository contains the code for the TRISAR pipeline built for the IEEE Data Fusion Contest.

Main workflow:
- download data
- train the model
- run the demo