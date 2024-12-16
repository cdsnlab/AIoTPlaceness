# Comprehensive Execution Manual for AIoTPlaceness

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Overview](#repository-overview)
3. [Prerequisites](#prerequisites)
   - System Requirements
   - Software Dependencies
4. [Directory Structure](#directory-structure)
5. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - Cloning the Repository
   - Setting Up the Environment
   - Dataset Preparation
   - Running Experiments
   - Visualization and Results
6. [Detailed Module Descriptions](#detailed-module-descriptions)
   - 2019 Social Activity Extractor
   - 2020 Enhancements
   - 2022 STARLAB Integration
   - 2024 Area Embeddings Project
7. [Troubleshooting and FAQs](#troubleshooting-and-faqs)
8. [Citation Guidelines](#citation-guidelines)
9. [Acknowledgments](#acknowledgments)

---

## Introduction

The **AIoTPlaceness** project is an ambitious initiative aimed at revolutionizing urban space management through advanced robotics and AI. By autonomously interpreting physical and social data, the project facilitates informed decision-making and predictions for smart cities. Key applications include:

- Urban Planning
- Smart City Infrastructure
- Traffic and Transportation Analytics

The repository encompasses multiple sub-projects spanning several years of development, each building upon the last to achieve greater functionality and precision.

---

## Repository Overview

The repository includes the following key components:

1. Algorithms for human activity recognition using multimodal data.
2. Machine learning models for urban space understanding.
3. Integrations of advanced neural architectures for spatial-temporal data processing.

### Key Contributions

- **Human Activity Recognition:** Leveraging semi-supervised learning for social data (e.g., Instagram).
- **Spatial-Temporal Modeling:** Employing graph convolutional recurrent networks (GCRNs).
- **Area Embeddings:** Novel embeddings for urban area characterization.

---

## Prerequisites

### System Requirements

- **Operating System:** Linux, macOS, or Windows (Unix-based systems recommended).
- **Memory:** Minimum 16 GB RAM (32 GB recommended for large-scale experiments).
- **Processor:** Multi-core processor with GPU support.
- **Disk Space:** At least 20 GB free space.

### Software Dependencies

- **Python Version:** Python 3.8 or higher.
- **Required Libraries:**
  - `numpy`
  - `pandas`
  - `torch`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `networkx`

- **Optional Tools for Visualization:**
  - `Geopandas`
  - `Plotly`

---

## Directory Structure

```
AIoTPlaceness-master/
|-- 2019
|-- 2020
|-- 2022_SWTEST_STARLAB
|-- 2024_AreaEmbeddings_Project
|-- LICENSE
|-- README.md
|-- ST-MFGCRN
```

### Key Directories and Their Contents

1. **2019/**: Contains early-stage implementations, including the social activity extractor.
2. **2020/**: Includes enhancements for multimodal data integration.
3. **2022_SWTEST_STARLAB/**: Collaboration with STARLAB for improved robotics integration.
4. **2024_AreaEmbeddings_Project/**: The latest developments focusing on area embeddings.
5. **ST-MFGCRN/**: Contains implementations of spatial-temporal graph convolutional recurrent networks.

---

## Step-by-Step Execution Guide

### Step 1: Cloning the Repository

Clone the repository locally:

```bash
git clone <repository-url>
cd AIoTPlaceness-master
```

### Step 2: Setting Up the Environment

Create a Python virtual environment to ensure an isolated setup:

```bash
python3 -m venv aiot_env
source aiot_env/bin/activate
pip install -r requirements.txt
```

If a `requirements.txt` file is missing, manually install dependencies:

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn networkx
```

### Step 3: Dataset Preparation

Datasets for experiments should be placed under the respective directories (`2019/`, `2020/`, etc.).

- Follow the dataset-specific instructions in each subdirectory.
- For `2024_AreaEmbeddings_Project/`, ensure raw data files are placed in the `data/` subfolder.

### Step 4: Running Experiments

Navigate to the desired sub-project directory and execute the respective script. For instance:

#### Example: Running the Area Embeddings Project

```bash
cd 2024_AreaEmbeddings_Project
python main.py --config config.json
```

### Step 5: Visualization and Results

Post-execution, visualize results using Jupyter Notebooks:

```bash
jupyter notebook
```

Open the appropriate notebook files in the `notebooks/` directory for interactive analysis.

---

## Detailed Module Descriptions

### 2019: Social Activity Extractor

- **Goal:** Human activity recognition using semi-supervised learning.
- **Key Script:** `extractor.py`
- **Notable Features:**
  - Instagram data processing.
  - Semi-supervised DEC model.

### 2020: Enhancements

- **Goal:** Multimodal data fusion for urban activity prediction.
- **Key Contributions:**
  - Integration of textual and spatial data.

### 2022: STARLAB Integration

- **Goal:** Robotics integration for smart cities.
- **Key Features:**
  - Real-time physical data interpretation.

### 2024: Area Embeddings Project

- **Goal:** Develop embeddings for urban area characterization.
- **Key Script:** `area_embedding.py`
- **Dataset Requirements:** Ensure high-resolution spatial-temporal data is preprocessed.

---

## Troubleshooting and FAQs

1. **Python Version Issues**:
   - Ensure Python 3.8 or higher is installed.

2. **Missing Dependencies**:
   - Run `pip install -r requirements.txt`.

3. **Data File Not Found**:
   - Verify file paths in configuration files.

---

## Citation Guidelines

If this repository contributes to your research, please cite appropriately:

```
@article{aiotplaceness2024,
  title={AIoTPlaceness: Robotics for Smart Cities},
  author={Author List},
  year={2024}
}
```

---

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II191126, Self-learning based Autonomic IoT Edge Computing).

---

# Related References

## 2019

### 2019/social-activity-extractor

* Dongmin Kim, Sumin Han, Heesuk Son, and Dongman Lee, “Human Activity Recognition using Semi-Supervised Multi-Modal DEC for Instagram Data”, PAKDD 2020.


### 2019/instagram-sampler

Helper web app for instagram categorization.


## 2020

### 2020/Gentrification

* Sumin Han, Dasom Hong, and Dongman Lee, “Exploring Commercial Gentrification using Instagram Data”, The 2020 IEEE/ACM International Conference on Advances in Social Network Analysis and Mining (ASONAM), December. 2020.

### 2020/HAR-instagram

Revised version for simplification of HAR on a Instagram post.

* Dongmin Kim, Sumin Han, Heesuk Son, and Dongman Lee, “Human Activity Recognition using Semi-Supervised Multi-Modal DEC for Instagram Data”, PAKDD 2020.
