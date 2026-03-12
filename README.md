# CDXML Converter

Utilities for working with ChemDraw `.cdxml` files.

This project can:

- convert `.cdxml` files into `SVG`, `PNG`, and `JPG`
- extract SMILES strings from CDXML files using RDKit
- build a simple image + SMILES dataset from a folder of CDXML files

## Project Structure

- `convert.py` - CLI for converting CDXML files to images
- `cdxml_converter/` - core parser, renderer, and rasterizer code
- `data_pipeline.py` - script to generate a dataset of images and SMILES
- `utility.ipynb` - notebook for ad hoc experimentation

## Requirements

- Python 3.10+
- `rdkit`
- `Pillow`
- `playwright` for high-quality PNG/JPG rasterization

If Playwright is not installed, the converter falls back to a Pillow-based renderer for raster image output.

## Install Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python packages you need:

```bash
pip install pillow rdkit playwright
python -m playwright install chromium
```

## Usage

### 1. Convert CDXML files

Convert one file to all supported output formats:

```bash
python convert.py input.cdxml
```

Convert specific formats only:

```bash
python convert.py input.cdxml -f svg png
```

Write outputs to a custom folder:

```bash
python convert.py input.cdxml -o output_dir
```

Batch convert multiple files:

```bash
python convert.py "*.cdxml"
```

Use custom rendering settings:

```bash
python convert.py input.cdxml --scale 3 --png-scale 3 --line-width 1.2 --font Arial
```

Common options:

- `-f, --formats` : output formats (`svg`, `png`, `jpg`)
- `-o, --output-dir` : output directory
- `--scale` : SVG rendering scale
- `--padding` : padding around the diagram
- `--bg-color` : background color
- `--png-scale` : raster output scale for PNG/JPG
- `--line-width` : override structure line width
- `--font` : SVG font family

The script reads the CDXML file with RDKit, extracts molecules, converts them to canonical SMILES, removes standalone `*` fragments, and prints the combined result.

### 2. Build a dataset

Update the `base` variable in `data_pipeline.py` so it points to the folder containing your `.cdxml` files, then run:

```bash
python data_pipeline.py
```

This script:

- creates `dataset/images/`
- converts each CDXML file into a PNG image
- extracts SMILES strings with RDKit
- removes images that do not produce valid SMILES
- writes `dataset.csv`

The generated CSV format is:

```csv
image,smiles
images/example.png,C1=CC=CC=C1
```

## Notes

- `convert.py` accepts glob patterns for batch processing.
- PNG and JPG generation is best when Playwright + Chromium are installed.
- `cdxml_to_smiles.py` and `data_pipeline.py` are currently written as simple scripts and require editing the file path variables before running.
# Chempanion
