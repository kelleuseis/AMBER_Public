# AMBER
AMBER contains labelled waveforms from microseismic events recorded on deep downhole sensor arrays, generated to stimulate research and development in the use of AI tools for downhole microseismic processing tasks. Raw SEGY waveforms can be converted into [Seisbench](https://github.com/seisbench/seisbench)-compatible datasets (<code>waveforms.hdf5</code> + <code>metadata.csv</code>) using the extraction pipeline <code>extract.py</code>. This repository also provides an event-centric PyTorch dataset with configurable downhole-specific augmentations for training deep-learning models on multi-station, multi-event waveforms.

AMBER has been compiled from 10 datasets (or sub-datasets):

| Dataset | Directory Name |
|-------|------|
Cotton Valley Stage B | CottonValley_StgB
Aneth CCS | Aneth
Clearfield MW4 monitoring well | Clearfield_mw4
Clearfield MW6 monitoring well | Clearfield_mw6
MSEEL stage 3H | MSEEL_3H
MSEEL stage 5H | MSEEL_5H
FORGE geothermal 2019 | FORGE_19
FORGE geothermal 2022 | FORGE_22
Preston New Road PNR-1 | PNR-1
Preston New Road PNR-2 | PNR-2

For inquiries, please contact:
james.verdon@bristol.ac.uk

## Installation

#### With pip
Install AMBER as a pip package
```bash
pip install .
```

#### With Conda
Create enviornment for reproducibility
```bash
conda env create -f env.yml
conda activate amber
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
The raw SEGY data is automatically downloaded as a [zip file](https://zenodo.org/records/18944111/files/amber_raw_segys.zip) to <code>AMBER_DATA_ROOT/raw</code> (<code>AMBER_DATA_ROOT</code> defaults to <code>~/.amber</code>, but can be configured via an environment variable) when <code>amber.database.extract_data()</code> is first run without a specified input zip filepath (if <code>is_zip</code> is set to <code>True</code> in <code>amber.database.BackendConfig</code>). Alternatively, users may manually download the raw data from [Zenodo](https://zenodo.org/records/18944111), and provide the zip filepath or its extracted contents to <code>amber.database.extract_data()</code> for custom conversion of the raw SEGY waveforms into Seisbench/AMBER-compatible datasets.

E.g. through <code>extract_cfg.yaml</code>:

```bash
backend_config:
    is_zip: True    # Extract from zip
    zipfilepath: null    # Zip filepath - amber_raw_segy.zip is auto-downloaded if set to None and default filepath (AMBER_DATA_RAW/amber_raw_segys.zip) does not exist
    eventdatadir: "${data_defaults.base_dir}/raw/EventWaveforms"   # Directory path containing event waveform SEGY sub-directories (used when is_zip is False)
    noisedatadir: "${data_defaults.base_dir}/raw/NoiseWaveforms"    # Directory path containing noise waveform SEGY sub-directories (used when is_zip is False)
    catalogdir: "${data_defaults.base_dir}/raw/Catalogs"    # Directory path containing event catalogue CSVs (used when is_zip is False)

data_config:
    outputdir: null    # Output directory for waveforms.hdf5 and metadata.csv - uses default AMBER_DATA_COMPILED path if set to None
    traintestsplit: 0.85    # Proportion of data [0-1] assigned to training
    trainvalsplit: 0.82    # Proportion of training data [0-1] used for training
    seed: 42    # For deterministic dataset generation
    datasets:
        - name: 'Aneth'    # Dataset name
          category: 'event'    # 'event' or 'noise' catalogue
          samplerate: 2000    # Sampling rate in Hz
          N: 1000    # Maximum number of samples to read in
          snrmin: 15    # Minimum median SNR from dataset for a useable event
          minppicks: 9    # Minimum number of P-wave picks from this dataset for a useable event
          minspicks: 9    # Minimum number of S-wave picks from this dataset for a useable event
          plen: 0.1    # Length of typical P-wave window for this dataset
          slen: 0.2    # Length of typical S-wave window for this dataset
          role: 'multi'    # "train", "test", or "dev" ("multi" for random splitting)
```

The raw SEGY zip file <code>amber_raw_segys.zip</code> has the following structure (<code>XXXX</code> is the dataset name):
```bash
< amber_raw_segys.zip >
   |
   |-- Attr/                         # Information about array positions and velocity models for each dataset
   |    |-- XXXX.Array.csv
   |    |-- XXXX.Vmodel.csv
   |
   |-- Catalogs/                     # Information about events for each dataset
   |    |-- XXXX.Catalog.csv
   |
   |-- EventWaveforms/               # Directories containing SEGY-format waveforms for the events of each dataset
   |
   |-- NoiseWaveforms/               # Directories containing SEGY-format event-free waveforms for each dataset
   |
   |-- ************************************************************************
```
<code>XXXX.Array.csv</code>: CSV files listing locations for each station in the array. Columns are: Station ID, Station X position, Station Y position, Station Depth. All are given in metres, normalised such that the deepest sensor in the string is at an X/Y position of [0,0].

<code>XXXX.Vmodel.csv</code>: CSV files listing a 1D layered velocity model for the site. Columns are: Layer ID, Layer top (in m), P-wave velocity, S-wave velocity (velocities in m/s)

<code>XXXX.Catalog.csv</code>: CSV files listing the events contained in each dataset. Columns are: Event ID, Data file containing the event, date of event, start time of traces in the data file, number of P-picks for the event, number of S-picks for the event, SNR for the event, The remaining columns list the P- and S-wave pick times at each station for the event (given in seconds from the trace start time (column 4))

Pre-generated Seisbench/AMBER-compatible <code>waveforms.hdf5</code> and <code>metadata.csv</code> are also available from Zenodo, and were produced using the executable script [extract.py](./extract.py) with the configuration file [extract_cfg.yaml](./extract_cfg.yaml):

```bash
python extract.py --config-path . --config-name extract_cfg
```

## Usage
See examples in [Examples.ipynb](./Examples.ipynb) for using the AMBER dataset for PyTorch training

## Acknowledgements
This work was funded by the [BOPS research consortium](https://www1.gly.bris.ac.uk/BOPS/), which is funded by a collection of hydrocarbon operating companies and service providers. None of these organisations had any input into the development, analysis or conclusions of this study.

## Reference
Leung, K., Lim, C.S.Y., Lapins, S., Read, E., Rodríguez-Pradilla, G., Verdon, J.P., Werner, M.J. The AI-Ready Downhole Microseismic Benchmark Database (AMBER). Seismological Research Letters, Under Review (2026)
