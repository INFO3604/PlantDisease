# PlantDisease

A comprehensive system for detecting and classifying plant diseases using deep learning.

## Project Structure

- **data/** - Raw, interim, processed datasets and train/val/test splits
- **models/** - Training checkpoints and exported models
- **reports/** - Figures and metrics outputs
- **scripts/** - Sequential pipeline scripts
- **src/plantdisease/** - Main package with configuration, data processing, and model code
- **webapp/** - Web application interface (Flask/FastAPI/Streamlit)
- **tests/** - Unit tests

## Quick Start

1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install: `pip install -r requirements.txt`
4. Run pipeline: `python scripts/00_download_data.py`, etc.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.
