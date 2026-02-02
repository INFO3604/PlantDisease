# PlantDisease

A mobile/web application for plant disease support can help farmers, agronomists, and gardening enthusiasts identify, manage, and prevent diseases affecting crops.

## Leaf Disease Detection using Computer Vision

This repository contains a Python-based machine learning pipeline for detecting plant leaf diseases using image processing and supervised learning techniques. The project focuses on analysing visible leaf symptoms, which are often linked to underlying plant health issues, including root and soilborne diseases.

## Repository Structure

- `notebooks/` – Jupyter notebooks used for experiments, analysis, and documentation  
- `data/` – datasets used in the project (large files are not tracked in GitHub)  
- `src/` – reusable Python modules for preprocessing, segmentation, and feature extraction  

## Tools and Technologies

- Python  
- Jupyter Notebook  
- OpenCV  
- scikit-learn  

## How to Run the Project

### Clone the repository

- git clone https://github.com/sonal/PlantDisease.git

- cd PlantDisease

### Create and activate a virtual environment

Windows (PowerShell):
- python -m venv venv
- venv\Scripts\activate

Mac/Linux:
- python -m venv venv
- source venv/bin/activate

### Install project dependencies
- pip install -r requirements.txt

### Launch Jupyter Notebook
- jupyter notebook

Once Jupyter opens in your browser, navigate to the notebooks folder and open:
- leaf_disease_detection.ipynb

- Run the notebook cells sequentially from top to bottom.
