'''
Function to extract Seisbench-compatible hdf5 and csv files for AMBER, 
and map-style PyTorch dataset designed for use with AMBER
'''

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import random, secrets
import h5py, obspy, json, re
from pathlib import Path

from datetime import timezone
from dataclasses import dataclass, field, asdict
from typing import List
from itertools import chain
from types import MappingProxyType

import logging, pprint


curdir = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,  # Default logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Standard format
)
logger = logging.getLogger(__name__)



@dataclass
class DataConfig:
    name: str    # Name of dataset (e.g. Aneth)
    category: str    # "noise" or "event"
    samplerate: int = 2000    # Sampling rate in Hz
    N: int = 5    # Maximum number of samples to read in
    snrmin: int = 15    # Minimum SNR from dataset for a useable event
    minppicks: int = 8    # Minimum number of P-wave picks from this dataset for a useable event
    minspicks: int = 8    # Minimum number of S-wave picks from this dataset for a useable event
    plen: float = 0.01   # Length of typical P-wave window for this dataset
    slen: float = 0.02   # Length of typical S-wave window for this dataset
    role: str = 'multi'    # "train", "test", or "dev" ("multi" for random splitting)
        
    def summary(self):
        pprint.pprint(asdict(self))
        
        
@dataclass
class MultiDataConfig:
    datasets: List[DataConfig] = field(default_factory=list)
    eventdatadir: str = f"{curdir / 'Data' / 'EventWaveforms'}"    # Directory containing SEGY event waveforms
    noisedatadir: str = f"{curdir / 'Data' / 'NoiseWaveforms'}"    # Directory containing SEGY noise waveforms
    catalogdir: str = f"{curdir / 'Data' / 'Catalogs'}"    # Directory containing event catalogs
    outputdir: str = f"{curdir / 'Data' / 'AMBER_Seisbench'}"    # Output directory for hdf5 and csv files
    max_nev: int = 3    # Maximum number of extra events to include when trace contains more than one
    windowlength: int = 2000    # Expected window length for model input
    traintestsplit: float = 0.85    # Proportion of data assigned to training
    trainvalsplit: float = 0.82    # Proportion of training data used for training
    eventsplit: bool = True    # Split dataset by events (not random) into train, test, and val
    seed: int = 100    # For deterministic dataset generation
        
    def summary(self):
        pprint.pprint(asdict(self))
        
        
@dataclass
class DatasetConfig:
    windowlength: int    # Target window length for model input
    nstation: int    # Target number of stations for model input
    normalisation: str = "stationwise"    # Choice of trace normalisation (stationwise, tracewise, eventwise)
    sequentialstations: bool = True   # Only extract sequential stations (vs random subset)
    fullphasecoverage: bool = False    # Extract event-centric windows with both p and s (vs phase-centric)
        
    def summary(self):
        pprint.pprint(asdict(self))


class AugmentationContext:
    '''
    Read-only context for augmentations requiring data outside of 
    current sample. Not to be mutated.
    '''
    def __init__(self, dataset, trace_grp, eventdf, stnout, window_idx, samplerate):
        self.dataset = dataset
        self.trace_grp = trace_grp
        self.eventdf = eventdf
        self.stnout = stnout
        self.window_idx = window_idx
        self.samplerate = samplerate

        
        
def extract_data(config: MultiDataConfig):
    '''
    Generate Seisbench-compatible hdf5 containing all traces (waveforms.hdf5),
    and metadata csv file (metadata.csv), from AMBER .segy files. Output 
    directory specified in config.
    
    Parameters
    ---------------------------------------------
    config: dataloaders.MultiDataConfig
    
    Output
    ---------------------------------------------
    None
    '''
    h5_path = Path(config.outputdir).expanduser() / "waveforms.hdf5"
    csv_path = Path(config.outputdir).expanduser() / "metadata.csv"
    metadata_rows = []
    failed_reads = []
    
    rng = (
        np.random.default_rng(config.seed)
        if config.seed is not None
        else np.random.default_rng()
    )
    
    with h5py.File(h5_path, "w") as h5f:
        traces_grp = h5f.require_group("data")
        metadata_grp = h5f.require_group("data_format")

        def add_metadata(datasetname, data):
            metadata_grp.create_dataset(
                datasetname,
                data=data,
                dtype=h5py.string_dtype(encoding="utf-8")
            )
            return None

        add_metadata("component_order", "NEZ")    # All traces are rotated NEZ
        add_metadata("dimension_order", "CW")

        buckets = []

        for cfg in config.datasets:
            data_config = DataConfig(**cfg)

            if len(config.datasets) > 1:
                logger.info(f"Extracting data from: {data_config.name}")
                
            is_event = data_config.category == "event"

            if is_event:
                catalog_path = (
                    Path(config.catalogdir).expanduser()
                    / f"{data_config.name}.Catalog.csv"
                )
                eventdf = pd.read_csv(catalog_path)

                
                # Selection of events based on SNR and number of available picks
                useable_mask = (
                    (eventdf["SNRmedian"] >= data_config.snrmin) &
                    (eventdf["NP"] >= data_config.minppicks) &
                    (eventdf["NS"] >= data_config.minspicks)
                )
                useable_indx = np.where(useable_mask)[0]

                if len(useable_indx) < data_config.N:
                    logger.warning(
                        f"Warning: number of useable events for {data_config.name} is {len(useable_indx)}, "
                        f"which is smaller than the requested {data_config.N}. Consider using more "
                        f"lenient criteria for pick numbers or SNR, or requesting fewer events"
                    )
                    used_indx = useable_indx
                else:
                    used_indx = rng.choice(
                        useable_indx,
                        size=data_config.N,
                        replace=False
                    )

                tp_cols = eventdf.columns[eventdf.columns.str.contains("TP")]
                ts_cols = eventdf.columns[eventdf.columns.str.contains("TS")]

                tp_start = eventdf[tp_cols].to_numpy()
                ts_start = eventdf[ts_cols].to_numpy()
                
                
            else:
                noiselist = list(
                    (Path(config.noisedatadir).expanduser() / data_config.name).glob("*.segy")
                )
                used_indx = rng.choice(
                    len(noiselist),
                    size=min(len(noiselist), data_config.N),
                    replace=False
                )

                
            ## SEGY Extraction
            for i in used_indx:
                if is_event:
                    segy_path = (
                        Path(config.eventdatadir).expanduser() / data_config.name / eventdf.loc[i, "DataFile"]
                    )
                else:
                    segy_path = noiselist[i]
                    hex_id = secrets.token_hex(5)

                try:
                    segydata = obspy.read(segy_path)
                except Exception as e:
                    failed_read = Path(segy_path.parent.name) / segy_path.name
                    failed_reads.append(failed_read)
                    logger.warning(f"Failed SEGY read ({failed_read}): {e}")
                    continue
                nstat = int(len(segydata) / 3)

                # Resample to ensure trace data does not need padding for expected window length
                wintime = segydata[0].stats.npts / segydata[0].stats.sampling_rate
                new_wintime = config.windowlength / data_config.samplerate

                if new_wintime > wintime:
                    min_newsamplerate = config.windowlength / wintime
                    k = int(np.ceil(min_newsamplerate / data_config.samplerate))
                    data_config.samplerate = k * data_config.samplerate
                    logger.debug(f"Resampling to {data_config.samplerate} Hz...")

                segydata.resample(data_config.samplerate)


                ndp = segydata[0].stats.npts

                if ndp not in buckets:
                    buckets.append(ndp)

                bucket_name = f"bucket{buckets.index(ndp)}"

                if bucket_name not in traces_grp:
                    dset = traces_grp.create_dataset(
                        bucket_name,
                        shape=(0, 3, ndp),
                        maxshape=(None, 3, ndp),
                        chunks=(1, 3, ndp),
                        dtype="float32",
                        compression="gzip"
                    )
                else:
                    dset = traces_grp[bucket_name]

                waves = np.array([trace.data for trace in segydata])
                waves = waves.reshape(
                    -1, 3, waves.shape[-1]
                )

                if is_event:
                    tp_primary = tp_start[i]
                    ts_primary = ts_start[i]

                    id_thisfile = np.where(
                        eventdf["DataFile"] == eventdf.loc[i, "DataFile"]
                    )[0]


                    id_thisfile = id_thisfile[id_thisfile != i]
                    tp_all = tp_start[id_thisfile]
                    ts_all = ts_start[id_thisfile]

                    tp = np.full((config.max_nev, nstat), np.nan)
                    ts = np.full((config.max_nev, nstat), np.nan)

                    nev = min(len(id_thisfile), config.max_nev)

                    tp[:nev] = tp_all[:nev]
                    ts[:nev] = ts_all[:nev]

                    def pickadjust(pickarr):
                        pickarr = pickarr 
                        return pickarr

                    tp_primary = tp_primary * data_config.samplerate
                    ts_primary = ts_primary * data_config.samplerate
                    tp = tp * data_config.samplerate
                    ts = ts * data_config.samplerate

                # tz-naive to tz-aware (UTC) (for avoiding ambiguity when loading into pandas)
                starttime = segydata[0].stats.starttime.datetime.replace(tzinfo=timezone.utc)

                for sta in range(nstat):               
                    idx = dset.shape[0]
                    dset.resize(idx + 1, axis=0)
                    dset[idx, :, :] = waves[sta]

                    trace_name = f"{bucket_name}${idx},:3,:{ndp}"

                    if is_event:
                        metadata_dict = {
                            "trace_name": trace_name,
                            "split": data_config.role,
                            "station": f"{sta:03d}",
                            "dataset": data_config.name.lower(),
                            "event_id": eventdf["EventID"][i].lower(),
                            "trace_sampling_rate_hz": data_config.samplerate,
                            "trace_category": "earthquake",
                            "trace_start_time": starttime.isoformat(),
                            "trace_p_arrival_sample": int(tp_primary[sta]) if ~np.isnan(tp_primary[sta]) else np.nan,
                            "trace_s_arrival_sample": int(ts_primary[sta]) if ~np.isnan(ts_primary[sta]) else np.nan,
                            "trace_p_length": data_config.plen,
                            "trace_s_length": data_config.slen,
                            "trace_component_order": "NEZ",
                            "dimension_order": "CW"
                        }

                        for evid in range(config.max_nev):
                            metadata_tp_key = f"trace_p{evid+2}_arrival_sample"
                            metadata_ts_key = f"trace_s{evid+2}_arrival_sample"
                            metadata_dict[metadata_tp_key] = int(tp[evid][sta]) if ~np.isnan(tp[evid][sta]) else np.nan
                            metadata_dict[metadata_ts_key] = int(ts[evid][sta]) if ~np.isnan(ts[evid][sta]) else np.nan

                    else:
                        metadata_dict = {
                            "trace_name": trace_name,
                            "split": data_config.role,
                            "station": f"{sta:03d}",
                            "dataset": data_config.name.lower(),
                            "event_id": f"{data_config.name.lower()}_noise_{hex_id}",
                            "trace_sampling_rate_hz": data_config.samplerate,
                            "trace_category": "noise",
                            "trace_start_time": starttime.isoformat(),
                            "trace_component_order": "NEZ",
                            "dimension_order": "CW"                            
                        }
                        
                    metadata_rows.append(metadata_dict)

        metadata_grp.create_dataset(
            "bucket_ndp",
            data=np.array(buckets, dtype=np.int32)
        )


    metadata_df = pd.DataFrame(metadata_rows)
    
    # Assign Train-Test-Validation split
    metadata_df_multi = metadata_df[metadata_df["split"] == "multi"].copy()
    
    p_test = 1 - config.traintestsplit
    p_val = config.traintestsplit * (1-config.trainvalsplit)
    p_train = config.traintestsplit * config.trainvalsplit
       
    if config.eventsplit:
        event_ids = metadata_df_multi["event_id"].unique()
        event_splits = rng.choice(
            ["train", "dev", "test"],
            size=len(event_ids),
            p=[p_train, p_val, p_test]
        )

        event_split_map = dict(zip(event_ids, event_splits))
        metadata_df_multi["split"] = metadata_df_multi["event_id"].map(event_split_map)
        
    else:
        metadata_df_multi["split"] = rng.choice(
            ["train", "dev", "test"],
            size=len(metadata_df_multi),
            p=[p_train, p_val, p_test]
        )

    metadata_df.loc[metadata_df_multi.index, "split"] = metadata_df_multi["split"]

    metadata_df.to_csv(csv_path, index=False)

    logger.warning("\n".join(str(r) for r in failed_reads))

    return None
    
    
    
    

class AMBER(Dataset):
    def __init__(self, config:DatasetConfig, h5_path, csv_path, labeller, augmentations=[], mode="train"):
        '''
        Multi-station, multi-pick PyTorch map-style dataset for the AMBER downhole 
        microseismic dataset. Generates waves (n_station, 3, n_datapoints) and 
        labels arrays from a single hdf5 file containing traces (n_channels, n_datapoints), 
        and a metadata csv file.
        
        Trace normalisation, event-centric random window extraction, and dtype conversions 
        are handled internally.
        
        Required columns for metadata csv (Seisbench-compatible):
        split
        event_id
        dataset
        trace_name (formatted 'bucket_name$row_idx,:n_chnl,:n_datapoints')
        trace_category ('earthquake' or 'noise')
        trace_sampling_rate_hz (in Hz)
        trace_p_length (in s)
        trace_s_length (in s)
        trace_p_arrival_sample
        trace_s_arrival_sample
        
        Parameters
        --------------------------------------------
        config: dataloaders.DatasetConfig
        
        h5_path: str or pathlib.Path
            relative/direct path to hdf5 file
        
        csv_path: str or pathlib.Path
            relative/direct path to csv file
            
        labeller: Labeller.labeller
            labeller to turn pick and metadata info into model 
            labels initiated through create_labeller via
            LabellerConfig
            
        augmentations: list of Augmentations.augmentation
            list of augmentations generated through 
            Augmentations.load_augmentations via AugmentationConfig
            
        mode: str
            chosen data split category
            Default: "train", "dev", "test"
            "all" for all data
            
        Outputs
        --------------------------------------------
        waves: torch.float32 tensor
        
        *labels : sequence of torch.float32 tensor
            one or more label tensors returned by the labeller -
            number, meaning, and shapes of labels are defined by 
            the labeller implementation.
        '''
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        
        self.config = config        
        self.h5_path = h5_path
        self._h5_file = None
        
        meta_all = pd.read_csv(csv_path)
        if mode == "all":
            self.meta = meta_all
        else:
            self.meta = meta_all[meta_all["split"] == mode]
            
        # Ensure metadata csv is not empty    
        if self.meta.empty:
            avail_cols = meta_all["split"].unique()
            self.logger.error(
                f"No rows found: '{mode}'"
                f"Available values: {avail_cols}"
            )
            raise ValueError(
                f"No rows found: '{mode}'"
                f"Available values: {avail_cols}"
            )
            

        # Ensure all events are compatible with required input dimensions
        self.events = {}
        self.noise_ids_dataset = {}
        self.event_ids_dataset = {}
        
        for eid, df in self.meta.groupby("event_id"):
            if len(df) < self.config.nstation:
                self.logger.warning(
                    f"Dropping event {eid}: "
                    f"n_station={len(df)} < required {self.config.nstation}"
                )
                continue

            ndp = int(df.iloc[0].trace_name.split(":", -1)[-1])
            if ndp < self.config.windowlength:
                self.logger.warning(
                    f"Dropping event {eid}: "
                    f"trace length {ndp} < windowlength {self.config.windowlength}"
                )
                continue
                
            ds = df["dataset"].iloc[0]
            if df["trace_category"].iloc[0] == "noise":
                self.noise_ids_dataset.setdefault(ds, []).append(eid)
            else:
                self.event_ids_dataset.setdefault(ds, []).append(eid)
                
            self.events[eid] = df.index.to_numpy()
            
        
        # Ensure dataset contains valid samples
        self.sample_ids = list(self.events.keys())
        if len(self.sample_ids) == 0:
            self.logger.error(
                "No valid samples left after filtering!"
            )
            raise RuntimeError(
                "No valid samples left after filtering!"
            )
            
                
        # Make event dictionaries read-only
        self.events = MappingProxyType(self.events)
        self.noise_ids_dataset = MappingProxyType(self.noise_ids_dataset)
        self.event_ids_dataset = MappingProxyType(self.event_ids_dataset)
        
        self.augmentations = augmentations
        self.labeller = labeller

        
    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_file["data"]       
        
        
    def extract_sample(self, sample_id, trace_grp):
        '''
        Retrieve trace data and metadata for specified event_id.
        trace_name in metadata must be named in format
        bucket_name$rowidx,:n_channel,:n_datapoints
        '''
        eventdf = self.meta.loc[self.events[sample_id]]
        
        trace_namedf = eventdf["trace_name"].str.split("$", expand=True)
        rowidxs = trace_namedf[1].str.split(",", expand=True)[0].astype(int).to_numpy()
        buckets = trace_namedf[0].to_numpy()
        
        # Ensure all traces are from the same bucket
        if not np.all(buckets == buckets[0]):
            self.logger.error(
                f"Inconsistent bucket detected in eventdf: "
                f"expected '{bucket_cache}', got '{bucket}' "
                f"at row {i}"
            )
            raise ValueError(
                f"Inconsistent bucket detected in eventdf: "
                f"expected '{bucket_cache}', got '{bucket}' "
                f"at row {i}"
            )
            
        waves = np.asarray(trace_grp[buckets[0]][rowidxs], dtype=np.float32)
        self.logger.debug(
            f"extract_sample - extracted waves shape: {waves.shape}"
        )        

        return waves, eventdf
        
        
    def extract_window(self, waves_raw, eventdf, stnout=None, samplerate=None, rng=None):
        '''
        Event-centric random window extraction from input raw data for 
        matching model inputs. Each raw data can only be cropped once
        to produce one window.
        '''
        nstat, _, ndp = waves_raw.shape
        
        if rng is None:
            rng = np.random
        
        # Select stations
        if stnout is None:
            if not self.config.sequentialstations:
                stnout = np.sort(
                    rng.choice(
                        np.arange(nstat),
                        size=self.config.nstation,
                        replace=False
                    )
                )
            else:
                stn_rng = np.arange(nstat - self.config.nstation + 1)
                stn_start = rng.choice(stn_rng)
                stnout = np.arange(
                    stn_start,
                    stn_start + self.config.nstation
                )


        waves = waves_raw[stnout].copy()    # Ensure original waves is not mutated
        
        raw_samplerate = eventdf['trace_sampling_rate_hz'].iloc[0]
        samplerate = raw_samplerate if samplerate is None else samplerate

        
        # Expected arrival column names
        tp_pattern = re.compile(r"^trace_[pP]\d*_arrival_sample$")
        ts_pattern = re.compile(r"^trace_[sS]\d*_arrival_sample$")

        tp_cols = [c for c in eventdf.columns if tp_pattern.match(c)]
        ts_cols = [c for c in eventdf.columns if ts_pattern.match(c)]
        tp_cols = sorted(tp_cols, key=lambda s: (len(s), s))
        ts_cols = sorted(ts_cols, key=lambda s: (len(s), s))
        
        if len(tp_cols) != len(ts_cols):
            self.logger.error("Possible mismatched P/S arrival columns in metadata!")
            raise ValueError("Possible mismatched P/S arrival columns in metadata!")
            
        
        pickarr = np.full((nstat, 2, len(tp_cols)), np.nan, dtype=np.float32)
        pickarr[:, 0, :] = eventdf[tp_cols].to_numpy(dtype=np.float32)
        pickarr[:, 1, :] = eventdf[ts_cols].to_numpy(dtype=np.float32)
        pickarr = pickarr[stnout]
        
        pickarr = pickarr * samplerate / raw_samplerate

        
        # Query if primary picks exist
        p_exists = np.any(~np.isnan(pickarr[:, 0, 0]))
        s_exists = np.any(~np.isnan(pickarr[:, 1, 0]))
        
        
        
        # Ensure waves are padded with zeros if shorter than config.windowlength
        if waves.shape[-1] < self.config.windowlength:
            waves = np.pad(
                waves,
                pad_width=((0, 0), (0, 0), (0, self.config.windowlength)),
                mode="constant",
                constant_values=0.0
            )            
            self.logger.warning(
                "Raw waveform is unexpectedly short! "
                "Padding applied - Consider changing raw waveform augmentation parameters?"
            )         
        
        
        # Random window
        if not (p_exists or s_exists):
            window_earliest = 0
            window_latest = max(ndp - self.config.windowlength, window_earliest)
        else:       
            if self.config.fullphasecoverage and (p_exists and s_exists):
                pick_earliest = np.nanmin(pickarr[:, 0, 0]) - eventdf['trace_p_length'].iloc[0]*samplerate
                pick_latest = np.nanmax(pickarr[:, 1, 0]) + eventdf['trace_s_length'].iloc[0]*samplerate
           
            else:
                phase = rng.choice(np.where([p_exists, s_exists])[0])
                phasecol = ['trace_p_length', 'trace_s_length']
                pick_earliest = np.nanmin(pickarr[:, phase, 0]) - eventdf[phasecol[phase]].iloc[0]*samplerate
                pick_latest = np.nanmax(pickarr[:, phase, 0]) + eventdf[phasecol[phase]].iloc[0]*samplerate
            
            window_earliest = pick_latest - self.config.windowlength
            window_latest = max(pick_earliest, window_earliest)

            
        window_idx = int(max(
            rng.random()*(window_latest - window_earliest) + window_earliest,
            0
        ))
        
        self.logger.debug(
            f"extract_window - chosen window_idx (range / max): {window_idx}"
            f" ({window_earliest} : {window_latest} / {self.config.windowlength})"
        ) 

        pickarr = pickarr - window_idx
        boundmask = (pickarr < 0) | (pickarr > self.config.windowlength)
        pickarr = pickarr.copy()
        pickarr[boundmask] = np.nan 
        
        waves = waves[:, :, window_idx:window_idx+self.config.windowlength]
        return waves, pickarr, stnout, window_idx
        
    
  
    def __getitem__(self, index):
        trace_grp = self._get_h5()        
        sid = self.sample_ids[index]
        self.logger.debug(
            f"Getting sample: {sid}"
        )
        
        # Separate sampling rng stream from augmentation stream
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            base_seed = torch.initial_seed() % 2**32
        else:
            base_seed = worker_info.seed % 2**32
            
        sample_seed = base_seed + index
        sample_rng = np.random.default_rng(sample_seed)
        

        waves_all, eventdf = self.extract_sample(sid, trace_grp)
        samplerate = None

        
        # Apply augmentations on raw data
        for aug in self.augmentations:
            if aug.scope == "raw":
                waves_all, samplerate = aug.augment_raw(waves_all, eventdf.copy(deep=False), samplerate)

        # Apply event/phase-centric random window cropping
        waves, pickarr, stnout, window_idx = self.extract_window(
            waves_all, eventdf, samplerate=samplerate, rng=sample_rng
        )

        # Apply augmentations on windowed data
        for aug in self.augmentations:
            if aug.scope == "windowed":
                waves, pickarr, stnout, augrequests = aug.augment_windowed(waves, pickarr, stnout)
                
                # Capture context for requests
                context = AugmentationContext(
                    dataset=self,
                    trace_grp=trace_grp,
                    eventdf=eventdf.copy(deep=False),
                    stnout=stnout.copy(),
                    window_idx=window_idx,
                    samplerate=samplerate
                )
                for req in augrequests:
                    waves, pickarr = req.apply(context, waves, pickarr)


        # Normalisation    
        norm = self.config.normalisation.lower()
        eps = 1e-8

        if norm == "stationwise":
            for k in range(waves.shape[0]):
                scale = np.max(np.abs(waves[k, :, :])) + eps
                waves[k, :, :] /= scale

        elif norm == "tracewise":
            for k in range(waves.shape[0]):
                for c in range(3):
                    scale = np.max(np.abs(waves[k, c, :])) + eps
                    waves[k, c, :] /= scale

        elif norm == "eventwise":
            waves /= (np.max(np.abs(waves)) + eps)

        else:
            self.logger.error(f"Normalisation '{self.config.normalisation}' not recognised")
            raise ValueError(
                f"Normalisation '{self.config.normalisation}' not recognised"
            )


        # Labeller
        labels = self.labeller(pickarr, eventdf, stnout, window_idx, samplerate)

        if not isinstance(labels, (list, tuple)):
            self.logger.error(
                "Labeller must return a list or tuple of label arrays"                
            )
            raise TypeError(
                "Labeller must return a list or tuple of label arrays"
            )

        # NumPy array to Torch tensor
        waves = torch.from_numpy(waves).float()

        labels = [
            torch.from_numpy(lbl).float()
            for lbl in labels
        ]

        return (waves, *labels)                
                
                
    def __len__(self):
        return len(self.sample_ids)
    
    
    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()
        
        
        
        
        
       
        
        
        
############# Visualization (for simple debugging)       
def plot_batch(waves=None, labels=None,
               only_waves=False, posplt=False,
               dataloader=None, batch_idx=0, 
               tol=1e-5, seq_pick=True, picklines=True, no_noise=False,
               argmax_pick=False, argmax_color=['red', 'blue', 'green']):
    if dataloader is not None:
        for i, data in enumerate(dataloader):
            if i == batch_idx:
                break

        if isinstance(data, dict):    # For Seisbench-compatibility
            waves = data.get("X", data.get("waves"))
            labels = data.get("y", data.get("labels"))
        elif isinstance(data, (list, tuple)):
            waves = data[0]
            labels = data[1] if len(data) > 1 else None
        else:
            raise TypeError("Unsupported dataloader output type.")
            
    elif any(x is None for x in [waves, labels]) and not only_waves:
        print('Input data insufficient for generating plots!')
        return None
    
    
    if waves is not None and only_waves:
        labels = torch.zeros_like(waves)
        labels[:, :, 2] = 1

    print(waves.shape, labels.shape)

    if len(waves.shape) == 3:
        waves = waves.unsqueeze(1)
        labels = labels.unsqueeze(1)
        
    if argmax_pick:
        labels = torch.argmax(labels, dim=2).unsqueeze(2)

    waves = waves.numpy()
    labels = labels.numpy()

    nbatch = waves.shape[0]
    nstation = waves.shape[1]


    total_subplots = nbatch * nstation + nbatch - 1
    fig, axes = plt.subplots(total_subplots, 1, figsize=(14, 0.5 * total_subplots + nbatch), 
                             sharex=True, gridspec_kw={'hspace': 0})
    
    if total_subplots == 1:
        axes = [axes]

    ax_idx = 0

    for j, event in enumerate(waves):
        for i, station_data in enumerate(event):
            ax_w = axes[ax_idx]  
            label_data = labels[j][i].squeeze()

            for chnl in range(3):              
                ax_w.plot(station_data[chnl, :])

            if not only_waves:
                if argmax_pick:
                    ylim = ax_w.get_ylim()
                    for k in range(len(argmax_color)):
                        ax_w.fill_between(
                            np.arange(label_data.size), 
                            ylim[0], ylim[1], 
                            where=(label_data == k), color=argmax_color[k], alpha=0.3
                        )
                elif seq_pick:
                    ax_w.plot(labels[j][i][0]*station_data[:3].max())
                    ax_w.plot(labels[j][i][1]*station_data[:3].max())
                    if picklines:
                        for pick_idx in np.where(labels[j][i][0] > 0.98)[0]:
                            ax_w.axvline(x=pick_idx, color='red', linestyle='--')
                        for pick_idx in np.where(labels[j][i][1] > 0.98)[0]:
                            ax_w.axvline(x=pick_idx, color='blue', linestyle='--')
                    if labels.shape[-2] == 3 and not no_noise:
                        ax_w.plot(labels[j][i][2]*station_data[:3].max())
                else:
                    for n_pick in range(labels.shape[-1]):
                        ax_w.axvline(x=labels[j][i][0][n_pick], color='red', linestyle='--')
                        ax_w.axvline(x=labels[j][i][1][n_pick], color='blue', linestyle='--')

            ax_w.tick_params(labelbottom=True)
            ax_idx += 1

        if j < nbatch - 1:
            axes[ax_idx].axis('off')
            ax_idx += 1

    plt.tight_layout(pad=1.0)

    plt.show()

    return None