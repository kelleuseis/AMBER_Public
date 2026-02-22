# AMBER
AMBER contains labelled waveforms from microseismic events recorded on deep downhole sensor arrays, generated to stimulate research and development in the use of AI tools for downhole microseismic processing tasks. Raw SEGY waveforms can be converted into Seisbench-compatible datasets (<code>waveforms.hdf5</code> + <code>metadata.csv</code>) using the extraction pipeline <code>extract.py</code>. This repository also provides an event-centric PyTorch dataset with configurable downhole-specific augmentations for training deep-learning models on multi-station, multi-event waveforms.

AMBER has been compiled from 10 datasets (or sub-datasets):

 - Cotton Valley Stage B `CottonValley_StgB`
 - Aneth CCS `Aneth`
 - Clearfield mw4 monitoring well `Clearfield_mw4`
 - Clearfield mw6 monitoring well `Clearfield_mw6`
 - MSEEL stage 3H `MSEEL_3H`
 - MSEEL stage 5H `MSEEL_5H`
 - FORGE geothermal 2019 `FORGE_19`
 - FORGE geothermal 2022 `FORGE_22`
 - Preston New Road PNR-1 `PNR-1`
 - Preston New Road PNR-2 `PNR-2`

For inquiries, please contact:
james.verdon@bristol.ac.uk

## Installation

#### With pip
Install AMBER as a pip package
```bash
pip install -e .
```

#### With Conda
Create enviornment for reproducibility
```bash
conda env create -f env.yml
```

## Structure
```bash
< PROJECT ROOT >
   |
   |-- src/          
   |    |-- amber/
   |         |-- __init__.py
   |         |-- Augmentations/           # Augmentation API
   |         |-- Labeller/                # Labeller API
   |         |-- database.py              # For compiling training data from raw SEGYs
   |         |-- dataloaders.py           # PyTorch dataset for AMBER
   |         |-- registry.py              # Registry to enable Hydra/YAML-driven configuration and documentation
   |         |-- utils.py                 # Helper functions
   |
   |-- tests/                             # Unit tests
   |
   |-- extract.py                         # Executable script to compile training data
   |-- extract_cfg.yaml                   # Configuration file for parameters used in extract.py 
   |
   |-- Examples.ipynb                     # Jupyter notebook showcasing use of AMBER via Seisbench and AMBER frameworks
   |
   |-- env.yml                            # ENV Configuration (conda)
   |-- pyproject.toml                     # Installation Configuration
   |
   |-- README.md                          # This file
   |-- .gitignore                         # gitignore
   |
   |-- ************************************************************************
```

## Data
The raw SEGY data is automatically downloaded as a [zip file](https://zenodo.org/records/18944111/files/amber_raw_segys.zip) to <code>AMBER_DATA_ROOT/raw</code> (<code>AMBER_DATA_ROOT</code> defaults to <code>~/.amber</code>, but can be configured via an environment variable) when <code>amber.database.extract_data()</code> is first run without a specified input zip filepath (if <code>is_zip</code> is set to <code>True</code> in <code>amber.database.BackendConfig</code>). Alternatively, users may manually download the raw data from [Zenodo](https://zenodo.org/records/18944111), and provide the zip filepath or its extracted contents to <code>amber.database.extract_data()</code>. 

Pre-generated Seisbench-compatible <code>waveforms.hdf5</code> and <code>metadata.csv</code> are also available from [Zenodo](https://zenodo.org/records/18944111/files/amber_seisbench.zip), and were produced using the executable script [extract.py](./extract.py) with the configuration file [extract_cfg.yaml](./extract_cfg.yaml).

## Usage
See examples in [Examples.ipynb](./Examples.ipynb)

## Acknowledgements
This work was funded by the [BOPS research consortium](https://www1.gly.bris.ac.uk/BOPS/), which is funded by a collection of hydrocarbon operating companies and service providers. None of these organisations had any input into the development, analysis or conclusions of this study.

## Reference
Leung, K., Lim, C.S.Y., Lapins, S., Read, E., Rodríguez-Pradilla, G., Verdon, J.P., Werner, M.J. The AI-Ready Downhole Microseismic Benchmark Database (AMBER). Seismological Research Letters, Under Review (2026)
