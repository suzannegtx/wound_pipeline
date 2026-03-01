# Wound Pipeline (demo)

Pipeline delivers ROI detection, binary wound mask, tissue segmentation, wound classification, metrics, overlays, and summary JSON.

## Install
```bash
python -m venv .venv
. .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python tools/download_dummy_weights.py  # creates dummy weights
