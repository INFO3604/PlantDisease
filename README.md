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
2. Activate (Unix): `source venv/bin/activate`
   Activate (Windows PowerShell): `venv\Scripts\Activate.ps1`
   Activate (Windows cmd): `venv\Scripts\activate.bat`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the demo preprocessing pipeline (example):

   ```bash
   c:/Users/robyn/PlantDisease/.venv/Scripts/python.exe scripts/demo_single_image.py --input data/demo_input --output data/demo_output
   ```

5. Notes about SAM (Segment Anything Model):
   - The pipeline uses a local SAM checkpoint if available at `models/sam_vit_b.pth`.
   - You may also set the `SAM_CHECKPOINT` environment variable to point to a custom path.
   - On machines without GPU the loader uses a CPU-friendly SAM configuration; for best performance use CUDA if available.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.
