# TRISAR: Self-Supervised Triplet Metric Learning for Temporal SAR Interpretation

This repository contains **TRISAR**, a self-supervised deep metric learning framework developed for the **2026 IEEE Data Fusion Contest**.

TRISAR is designed for temporal SAR image analysis, with a focus on learning robust patch-level representations for temporal comparison, suspicious region ranking, timeline-based interpretation, compact localization of suspicious regions, and interactive visual exploration through a demo application.

## Authors

- **Jamil J. Ghazal**
- **Vera Könyves**
- **András Jung**


## Overview

TRISAR is a self-supervised deep metric learning framework for temporal SAR interpretation in dense and heterogeneous image time series. The method is designed for scenarios where repeated SAR observations of the same geographic region may differ not only because of real physical scene evolution, but also because of nuisance factors such as speckle, radiometric variability, acquisition geometry, residual misregistration, and short-lived imaging artifacts.

Instead of performing unstable direct image-domain differencing, TRISAR learns a discriminative embedding space with a shared deep neural encoder trained on anchor-positive-negative SAR triplets. The network is optimized to pull temporally corresponding observations closer in feature space while pushing spatially or temporally inconsistent observations farther apart. This makes the learned representation substantially more robust for temporal reasoning than raw intensity comparison alone.

The framework combines a ConvNeXt-based backbone, SAR-oriented augmentation, spatially harder negative mining, and hybrid metric-learning objectives to learn temporally stable local descriptors without requiring dense pixel-level change annotations. Once trained, the model supports several downstream analysis tasks, including suspicious temporal pair ranking, embedding-based timeline analysis, compact suspicious-region localization, and interactive visual inspection through the demo application.

From a methodological perspective, TRISAR can be understood as a representation-learning-based alternative to conventional change detection pipelines. Rather than directly predicting binary change masks, it learns a compact semantic similarity structure over repeated SAR observations, which can then be used for scalable screening, transient-versus-persistent temporal interpretation, and spatially interpretable anomaly discovery.

This repository contains the full research workflow behind TRISAR, including data download and preparation, self-supervised training, model inference, timeline-based analysis, suspicious-region localization, and an interactive demo for qualitative exploration of temporal SAR behavior.

## Main Features

- **Self-supervised triplet metric learning** for SAR patch representations
- **Temporal analysis** of local regions across multiple acquisition dates
- **Suspicious pair ranking** based on learned embedding distances
- **Temporary change analysis** for short-lived anomalous events
- **Continuous change analysis** for gradual or persistent scene evolution
- **Compact localization** of suspicious regions
- **Interactive Streamlit demo** for exploring scenes, timelines, and localization outputs



## Installation

Clone the repository and install the required packages:
```bash
git clone https://github.com/GhazalJamil/TRISAR.git

cd TRISAR
pip install -r requirements.txt
```

## Workflow

This repository follows a simple three-step workflow.

### 1. Download and Prepare Data

First, download the SAR data and prepare the required inputs:

- Open and run `data/download.ipynb`
- Filter and download SAR TIFF files near predefined locations
- Organize the downloaded scenes into the expected folder structure
- Prepare patch datasets and CSV files needed for training and analysis

This step creates the temporal SAR patch collections used by the model and the demo application.

### 2. Train a Model or Use a Pre-trained Checkpoint

You can either train TRISAR from scratch or use the provided pre-trained model.

#### Option A — Train Your Own Model

- Open and run `train/train.ipynb`
- Train the TRISAR model on the prepared SAR patch dataset
- Save the trained checkpoint
- Use the resulting `.pt` file in the demo application

#### Option B — Use the Pre-trained Model

- Download the trained checkpoint from Google Drive:
  https://drive.google.com/file/d/1sGaRBHGmmxrBw1UJQNk07qaqNNAitMyL/view?usp=drive_link
- Place the checkpoint in the appropriate folder
- Load it in the demo application

### 3. Run the Demo Application

To launch the interactive demo:

streamlit run demo/trisar_app.py

Inside the interface, you can:
- load a trained model checkpoint,
- inspect downloaded TIFF scenes,
- analyze patch timelines,
- visualize temporary and continuous changes,
- localize suspicious regions,
- explore model-based temporal interpretation results.

### For more INFORMATION check: demo.md



## Acknowledgment

This work was prepared for the **2026 IEEE GRSS Data Fusion Contest**.

The authors would like to thank the **IEEE Geoscience and Remote Sensing Society (GRSS)**, the **Image Analysis and Data Fusion (IADF) Technical Committee**, and **Capella Space** for organizing the contest and providing the SAR data used in this work.

## Contact

For questions, collaboration, or further information, please contact:

**Jamil József Ghazal**
GitHub: https://github.com/GhazalJamil
Email: gy51by@inf.elte.hu

You may also open an issue in this repository for project-related questions.


## Citation

If you use this repository, please cite this codebase.

```bibtex
@misc{ghazal2026trisar,
  title={TRISAR: Self-Supervised Triplet Metric Learning for Temporal SAR Interpretation},
  author={Ghazal, Jamil J{\'o}zsef and Jung, Andr{\'a}s and K{\"o}nyves, Vera},
  year={2026},
  note={Code available at: https://github.com/GhazalJamil/TRISAR}
}