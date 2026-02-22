'''Map-style PyTorch dataset designed for use with AMBER'''

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import h5py, re
from pathlib import Path

from dataclasses import dataclass, field, asdict
from itertools import chain
from types import MappingProxyType

import logging, pprint

from .utils import check_if_exists
        
        
@dataclass(frozen=True)
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





class AMBER(Dataset):
    def __init__(self, config:DatasetConfig, h5_path, csv_path, labeller, augmentations=None, mode="train"):
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.config = config        
        self.h5_path = check_if_exists(h5_path, self.logger)
        self._h5_file = None

        meta_all = pd.read_csv(check_if_exists(csv_path, self.logger))

        # Check schema
        required_cols = {
            "split", "event_id", "dataset",
            "trace_name", "trace_category",
            "trace_sampling_rate_hz",
            "trace_p_arrival_sample",
            "trace_s_arrival_sample",
            "trace_p_length", "trace_s_length"
        }

        missing_cols = required_cols - set(meta_all.columns)

        if missing_cols:
            liststr = "\n".join(map(str, sorted(missing_cols)))
            msg = f"Metadata csv missing required columns:\n{liststr}"
            self.logger.error(msg)
            raise ValueError(msg)


        if mode == "all":
            self.meta = meta_all
        else:
            self.meta = meta_all[meta_all["split"] == mode]
            
        # Ensure metadata csv is not empty    
        if self.meta.empty:
            avail_cols = meta_all["split"].unique()
            msg = (
                f"No rows found: '{mode}'"
                f"Available values: {avail_cols}"            
            )
            self.logger.error(msg)
            raise ValueError(msg)
            

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
                
            ds = df["dataset"].iloc[0]
            if df["trace_category"].iloc[0] == "noise":
                self.noise_ids_dataset.setdefault(ds, []).append(eid)
            else:
                self.event_ids_dataset.setdefault(ds, []).append(eid)
                
            self.events[eid] = df.index.to_numpy()
            
        
        # Ensure dataset contains valid samples
        self.sample_ids = list(self.events.keys())
        if len(self.sample_ids) == 0:
            msg = "No valid samples left after filtering!"
            self.logger.error(msg)
            raise RuntimeError(msg)
            
                
        # Make event dictionaries read-only
        self.events = MappingProxyType(self.events)
        self.noise_ids_dataset = MappingProxyType(self.noise_ids_dataset)
        self.event_ids_dataset = MappingProxyType(self.event_ids_dataset)
        
        self.augmentations = list(augmentations) if augmentations is not None else []
        self.labeller = labeller

        
    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_file["data"]       

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state
        
        
    def extract_sample(self, sample_id, trace_grp):
        '''
        Retrieve trace data and metadata for specified event_id.
        trace_name in metadata must be named in format
        bucket_name$rowidx,:n_channel,:n_datapoints
        '''
        eventdf = self.meta.loc[self.events[sample_id]]
        
        trace_namedf = eventdf["trace_name"].str.split("$", expand=True)

        if trace_namedf.shape[1] != 2:
            msg = (
                "trace_name format invalid. Expected format: "
                "'bucket_name$rowidx,:n_channel,:n_datapoints'"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        row_namedf = trace_namedf[1].str.split(",", expand=True)
        try:
            rowidxs = row_namedf[0].astype(int).to_numpy()
        except Exception:
            msg = "trace_name row index is not an integer!"
            self.logger.error(msg)
            raise ValueError(msg)

        buckets = trace_namedf[0].to_numpy()
        
        
        # Ensure all traces are from the same bucket
        if not np.all(buckets == buckets[0]):
            msg = (
                f"Inconsistent bucket detected in eventdf: "
                f"expected '{bucket_cache}', got '{bucket}' "
                f"at row {i}"            
            )
            self.logger.error(msg)
            raise ValueError(msg)
            
        waves = np.asarray(trace_grp[buckets[0]][rowidxs], dtype=np.float32)
        
        if len(waves.shape) != 3 or waves.shape[1] != 3:
            msg = (
                f"extracted waves shape: {waves.shape} - "
                "waves must be in shape (n_sta, 3, n_datapoints)!"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return waves, eventdf
        
        
    def extract_window(self, waves_raw, eventdf, stnout=None, samplerate=None, rng=None):
        '''
        Event-centric random window extraction from input raw data for 
        matching model inputs. Each raw data can only be cropped once
        to produce one window.
        '''
        nstat_raw, _, ndp_raw = waves_raw.shape
        
        if rng is None:
            rng = np.random
        
        # Select stations
        if stnout is None:
            if not self.config.sequentialstations:
                stnout = np.sort(
                    rng.choice(
                        np.arange(nstat_raw),
                        size=self.config.nstation,
                        replace=False
                    )
                )
            else:
                stn_rng = np.arange(nstat_raw - self.config.nstation + 1)
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
            msg = "Possible mismatched P/S arrival columns in metadata!"
            self.logger.error(msg)
            raise ValueError(msg)
            
        
        pickarr = np.full((nstat_raw, 2, len(tp_cols)), np.nan, dtype=np.float32)
        pickarr[:, 0, :] = eventdf[tp_cols].to_numpy(dtype=np.float32)
        pickarr[:, 1, :] = eventdf[ts_cols].to_numpy(dtype=np.float32)
        pickarr = pickarr[stnout]
        
        pickarr *= np.float32(samplerate / raw_samplerate)

        
        # Query if primary picks exist
        p_exists = np.any(~np.isnan(pickarr[:, 0, 0]))
        s_exists = np.any(~np.isnan(pickarr[:, 1, 0]))
        
        
        
        # Ensure waves are padded with zeros if shorter than config.windowlength
        if ndp_raw < self.config.windowlength:
            waves = np.pad(
                waves,
                pad_width=((0, 0), (0, 0), (0, self.config.windowlength)),
                mode="constant",
                constant_values=0.0
            )            
            self.logger.warning(
                f"{eventdf['event_id'].iloc[0]}: Raw waveform is unexpectedly short! "
                "Padding applied - Consider changing raw waveform augmentation parameters? "
                f"wave_ndp|new_wave_ndp|config_ndp: {ndp_raw}|{waves.shape[-1]}|{self.config.windowlength}"
            )         
        
        
        # Random window
        window_latest_max = waves.shape[-1] - self.config.windowlength
        
        if not (p_exists or s_exists):
            window_earliest = 0
            window_latest = max(window_latest_max, window_earliest)
        else:       
            if self.config.fullphasecoverage and (p_exists and s_exists):
                pick_earliest = np.nanmin(pickarr[:, 0, 0]) - eventdf['trace_p_length'].iloc[0]*samplerate
                pick_latest = np.nanmax(pickarr[:, 1, 0]) + eventdf['trace_s_length'].iloc[0]*samplerate
           
            else:
                phase = rng.choice(np.where([p_exists, s_exists])[0])
                phasecol = ['trace_p_length', 'trace_s_length']
                pick_earliest = np.nanmin(pickarr[:, phase, 0]) - eventdf[phasecol[phase]].iloc[0]*samplerate
                pick_latest = np.nanmax(pickarr[:, phase, 0]) + eventdf[phasecol[phase]].iloc[0]*samplerate
            
            window_earliest = max(min(pick_latest - self.config.windowlength, window_latest_max), 0)
            window_latest = min(max(pick_earliest, window_earliest), window_latest_max)
            
        window_idx = int(rng.random()*(window_latest - window_earliest) + window_earliest)
        
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
            msg = f"Normalisation '{self.config.normalisation}' not recognised"
            self.logger.error(msg)
            raise ValueError(msg)


        # Labeller
        labels = self.labeller(pickarr, eventdf, stnout, window_idx, samplerate)

        if not isinstance(labels, (list, tuple)):
            msg = "Labeller must return a list or tuple of label arrays"
            self.logger.error(msg)
            raise TypeError(msg)

        # NumPy array to Torch tensor
        waves = torch.from_numpy(waves).float()

        labels = [
            torch.from_numpy(lbl).float()
            for lbl in labels
        ]

        return (waves, *labels)                
                
                
    def __len__(self):
        return len(self.sample_ids)
    
    
    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
