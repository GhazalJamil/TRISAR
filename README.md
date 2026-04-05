# TRISAR: Self-Supervised Triplet Metric Learning for Temporal SAR Interpretation

This repository was created for the 2026 IEEE Data Fusion Contest.
Authors:
- Jamil J Ghazal
- Vera Könyves
- András Jung

## Installation

Install the required packages with:

pip install -r requirements.txt

# Steps to follow:

This document outlines the steps to use the TRISAR project for temporal SAR interpretation.

## Step 1: Download Data

First, download the required SAR data using the download notebook:

- Open and run `data/download.ipynb`
- This will filter and download SAR TIFF files near predefined locations
- Organize the data into appropriate folders
- Prepare patch datasets and CSV files

## Step 2: Train or Use Pre-trained Model

You have two options for obtaining a model:

### Option A: Train Your Own Model

- Open and run `train/train.ipynb`
- This will train the TRISAR model using the downloaded data
- Save the trained checkpoint
- Use it inside demo app (step 3.)

### Option B: Use Pre-trained Model

- Download our trained model from Google Drive: [link](link)
- Place the checkpoint in the appropriate location
- Use it inside demo app (step 3.)

## Step 3: Run the Demo Application

- Start the demo application: `streamlit run demo/trisar_app.py`
- Use your model (.pt) file, -> UI -> Model -> Checkpoint path
- Use the demo for:
  - Inspecting downloaded TIFF scenes
  - Analyzing patch timelines
  - Visualizing temporal and continuous changes
  - Localizing suspicious regions
  - Change detection
  - Timeline analysis

You can pass command-line parameters to customize the demo behavior.