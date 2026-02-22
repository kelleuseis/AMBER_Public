'''Functions to extract Seisbench-compatible hdf5 and csv files for AMBER'''

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import h5py, json, zipfile
import secrets, os, io
from pathlib import Path
from obspy.io.segy.segy import SEGYFile
from contextlib import contextmanager

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional

import logging, pprint

from amber import AMBER_DATA_RAW, AMBER_DATA_COMPILED, ensure_dataset
from .utils import check_if_exists


logger = logging.getLogger(__name__)



@dataclass(frozen=True)
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
        
        
@dataclass(frozen=True)
class MultiDataConfig:
    datasets: Tuple[DataConfig] = field(default_factory=list)
    outputdir: str = None    # Output directory for hdf5 and csv files
    max_nsev: int = 3    # Maximum number of secondary events to include when trace contains more than one
    traintestsplit: float = 0.85    # Proportion of data [0-1] assigned to training
    trainvalsplit: float = 0.82    # Proportion of training data [0-1] used for training
    eventsplit: bool = True    # Split dataset by events (not random) into train, test, and val
    seed: int = 42    # For deterministic dataset generation
    batchsize: int = 10000    # Batch size for writing to csv (in n_traces)
        
    def summary(self):
        pprint.pprint(asdict(self))

@dataclass(frozen=True)
class BackendConfig:
    is_zip: bool = True    # Extract from zip
    zipfilepath: Optional[str] = None    # zip filepath containing all directories
    eventdatadir: Optional[str] = f"{AMBER_DATA_RAW / 'EventWaveforms'}"    # Directory containing SEGY event waveforms
    noisedatadir: Optional[str] = f"{AMBER_DATA_RAW / 'NoiseWaveforms'}"    # Directory containing SEGY noise waveforms
    catalogdir: Optional[str] = f"{AMBER_DATA_RAW / 'Catalogs'}"    # Directory containing event catalogs
    datasetname: str = "amber_default"
    autodownload: bool = True
    
    
class BackendWrapper:
    '''Wrapper for connecting extraction function to database'''
    def __init__(self, config: BackendConfig):
        '''
        Attributes
        ------------------------------------
        eventdatadir: pathlib.Path
            path within zip file or absolute/relative path
            to directory containing event SEGY traces
            
        noisedatadir: pathlib.Path
            path within zip file or absolute/relative path
            to directory containing noise SEGY traces
            
        catalogdir: pathlib.Path
            path within zip file or absolute/relative path
            to directory containing event metadata csv            
        '''
        if config.is_zip:
            if config.zipfilepath is None:
                zipfilepath = ensure_dataset(config.datasetname, config.autodownload)
            else:
                zipfilepath = check_if_exists(config.zipfilepath, logger)
            
            self.z = zipfile.ZipFile(zipfilepath, 'r')
            self.eventdatadir = Path('EventWaveforms')
            self.noisedatadir = Path('NoiseWaveforms')
            self.catalogdir = Path('Catalogs')
            
        else:
            self.z = None
            self.eventdatadir = check_if_exists(config.eventdatadir, logger)
            self.noisedatadir = check_if_exists(config.noisedatadir, logger)
            self.catalogdir = check_if_exists(config.catalogdir, logger)
            
    def extract(self, path):
        '''
        Return a readable, seekable binary file-like object for a file
        within zip or a normal directory
        
        Parameters
        -------------------------------------------
        path: str or pathlib.Path
        '''
        if self.z is not None:
            return io.BytesIO(self.z.read(str(path)))
        else:
            return open(path, "rb")
        
    def return_filelist(self, dirname, ext):
        '''
        Return a list of pathlib paths within a directory matching 
        a given file extension within zip or a normal directory
        
        Parameters
        --------------------------------------------
        dirname: str or pathlib.Path
            directory path
        ext: str
            extension string
        '''
        if self.z is not None:
            filelist = self.z.namelist()
            return [
                Path(f) for f in filelist
                if f.startswith(str(dirname))
                and f.endswith(ext)
                and not os.path.basename(f).startswith("._")
                # ignore files starting with '._' created by macOS as part of system handling
            ]
        
        else:
            return list(Path(dirname).glob(f"*{ext}"))
    
    def close(self):
        if self.z is not None:
            self.z.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    
    


def extract_data(config: MultiDataConfig, be_config: BackendConfig):
    '''
    Function to allow for different dataset compositions (multi-dataset train/test splits,
    snr filtering, minimum pick filtering). Generates Seisbench-compatible hdf5 containing 
    all traces (waveforms.hdf5), and metadata csv file (metadata.csv), from AMBER .segy 
    files. Output directory specified in config (default AMBER_DATA_COMPILED). All traces 
    are by default NEZ rotated, and in dimensions (n_chnl, n_datapoints).
    
    Each dataset sample corresponds to a single event occurring within a waveform trace. If 
    multiple events are present in the same trace, the trace is reused to create multiple 
    samples, each centered on a different event (primary event).
    
    
    
    Metadata columns
    --------------------------------------------
    trace_name: str
        bucket_name$row_idx,:n_chnl,:n_datapoints
    
    split: str
        'train', 'test', or 'dev'
        
    station: str
        station index (001, 002, 003, ...)
        
    dataset: str
        dataset name
        
    event_id : str
        event identifier associated with this sample
        
    trace_sampling_rate_hz: int
        sampling rate in Hz
        
    trace_category: str
        'earthquake' or 'noise'
    
    trace_p_arrival_sample : int
        sample index from trace start of the P-wave arrival 
        for the primary event (event_id)

    trace_s_arrival_sample : int
        sample index from trace start of the S-wave arrival 
        for the primary event (event_id)

    trace_p{n}_arrival_sample : int
        sample index from trace start of the P-wave arrival 
        for the nth secondary event

    trace_s{n}_arrival_sample : int
        sample index from trace start of the S-wave arrival 
        for the nth secondary event
        
    trace_start_time: str
        iso format datetime string (UTC tz-aware)
        
    trace_p_length: float
        protection buffer duration (seconds) around the primary
        P-wave arrival used to constrain random window cropping
        
    trace_s_length: float
        protection buffer duration (seconds) around the primary
        S-wave arrival used to constrain random window cropping
        
    trace_component_order: 'NEZ'
        
    dimension_order: 'CW'
    
    
    Parameters
    ---------------------------------------------
    config: database.MultiDataConfig
    be_config: database.BackendConfig
    
    Output
    ---------------------------------------------
    None
    '''
    if config.outputdir is None:
        AMBER_DATA_COMPILED.mkdir(parents=True, exist_ok=True)
        outputdir = AMBER_DATA_COMPILED
    else:
        outputdir = check_if_exists(config.outputdir, logger)
        
    h5_path = outputdir / "waveforms.hdf5"
    csv_path = outputdir / "metadata.csv"
    
    if h5_path.exists() or csv_path.exists():
        msg = (
            "hdf5 or csv file already exists in designated output directory! "
            f"outputdir: {outputdir}"
        )
        logger.error(msg)
        raise RuntimeError(msg)
    
    batch_rows = []
    nsamples = 0
    header = True
    failed_reads = []
    
    rng = (
        np.random.default_rng(config.seed)
        if config.seed is not None
        else np.random.default_rng()
    )
    
    with BackendWrapper(be_config) as be, \
         h5py.File(h5_path, "w") as h5f:
        traces_grp = h5f.require_group("data")
        metadata_grp = h5f.require_group("data_format")

        def add_metadata(datasetname, data):
            metadata_grp.create_dataset(
                datasetname,
                data=data,
                dtype=h5py.string_dtype(encoding="utf-8")
            )
            return None

        add_metadata("component_order", "NEZ")
        add_metadata("dimension_order", "CW")

        buckets = []
        
        # schema
        basecols = [
            "trace_name",
            "split",
            "station",
            "dataset",
            "event_id",
            "trace_sampling_rate_hz",
            "trace_category",
            "trace_start_time",
            "trace_p_arrival_sample",
            "trace_s_arrival_sample",
            "trace_p_length",
            "trace_s_length",
            "trace_component_order",
            "dimension_order"
        ]
        seccols = []
        for evid in range(config.max_nsev):
            seccols.append(f"trace_p{evid+2}_arrival_sample")
            seccols.append(f"trace_s{evid+2}_arrival_sample")

        metacols = basecols + seccols
        
        def dict2csv(batch_rows, header):
            batch_df = pd.DataFrame(batch_rows).assign(
                trace_component_order="NEZ",
                dimension_order="CW"
            ).reindex(columns=metacols, fill_value=np.nan)
            batch_df.to_csv(csv_path, mode="a", header=header, index=False)
            return None
        

        for cfg in config.datasets:
            data_config = DataConfig(**cfg)

            if len(config.datasets) > 1:
                logger.info(f"Extracting data from: {data_config.name}")
                
            is_event = data_config.category == "event"

            if is_event:
                eventdf = pd.read_csv(be.extract(
                    be.catalogdir / f"{data_config.name}.Catalog.csv"
                ))

                
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
                noiselist = be.return_filelist(
                    be.noisedatadir / data_config.name, ".segy"
                )
                used_indx = rng.choice(
                    len(noiselist),
                    size=min(len(noiselist), data_config.N),
                    replace=False
                )

                
            ## SEGY Extraction
            for i in used_indx:
                if is_event:
                    segy_path = be.eventdatadir / data_config.name / eventdf.loc[i, "DataFile"]
                else:
                    segy_path = noiselist[i]
                    hex_id = secrets.token_hex(5)

                try:
                    with be.extract(segy_path) as s:
                        segydata = SEGYFile(s)
                        nstat = int(len(segydata.traces) / 3)
                        segydata_samplerate = 1e6 / segydata.binary_file_header.sample_interval_in_microseconds
                        waves = np.array([tr.data for tr in segydata.traces])
                        
                        hdr = segydata.traces[0].header
                        # tz-aware (UTC) (for avoiding ambiguity when loading into pandas)
                        starttime = datetime(hdr.year_data_recorded, 1, 1, tzinfo=timezone.utc) \
                                    + timedelta(
                                        days=hdr.day_of_year-1,
                                        hours=hdr.hour_of_day,
                                        minutes=hdr.minute_of_hour,
                                        seconds=hdr.second_of_minute
                                    )

                except Exception as e:
                    failed_read = Path(segy_path.parent.name) / segy_path.name
                    failed_reads.append(failed_read)
                    logger.warning(f"Failed SEGY read ({failed_read}): {e}")
                    continue

                waves = waves.reshape(-1, 3, waves.shape[-1])
                samplerate = data_config.samplerate
                
                scale = samplerate / segydata_samplerate
                waves = resample_poly(
                    waves, int(round(scale*1000)), 1000, axis=-1
                )
                ndp = waves.shape[-1]

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


                if is_event:
                    tp_primary = tp_start[i]
                    ts_primary = ts_start[i]

                    id_thisfile = np.where(
                        eventdf["DataFile"] == eventdf.loc[i, "DataFile"]
                    )[0]


                    id_thisfile = id_thisfile[id_thisfile != i]
                    tp_all = tp_start[id_thisfile]
                    ts_all = ts_start[id_thisfile]

                    tp = np.full((config.max_nsev, nstat), np.nan)
                    ts = np.full((config.max_nsev, nstat), np.nan)

                    nev = min(len(id_thisfile), config.max_nsev)

                    tp[:nev] = tp_all[:nev]
                    ts[:nev] = ts_all[:nev]

                    def pickadjust(pickarr):
                        pickarr = pickarr 
                        return pickarr

                    tp_primary = tp_primary * samplerate
                    ts_primary = ts_primary * samplerate
                    tp = tp * samplerate
                    ts = ts * samplerate


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
                            "trace_sampling_rate_hz": samplerate,
                            "trace_category": "earthquake",
                            "trace_start_time": starttime.isoformat(),
                            "trace_p_arrival_sample": int(tp_primary[sta]) if not np.isnan(tp_primary[sta]) else np.nan,
                            "trace_s_arrival_sample": int(ts_primary[sta]) if not np.isnan(ts_primary[sta]) else np.nan,
                            "trace_p_length": data_config.plen,
                            "trace_s_length": data_config.slen
                        }

                        for evid in range(config.max_nsev):
                            metadata_tp_key = f"trace_p{evid+2}_arrival_sample"
                            metadata_ts_key = f"trace_s{evid+2}_arrival_sample"
                            metadata_dict[metadata_tp_key] = int(tp[evid][sta]) if not np.isnan(tp[evid][sta]) else np.nan
                            metadata_dict[metadata_ts_key] = int(ts[evid][sta]) if not np.isnan(ts[evid][sta]) else np.nan

                    else:
                        metadata_dict = {
                            "trace_name": trace_name,
                            "split": data_config.role,
                            "station": f"{sta:03d}",
                            "dataset": data_config.name.lower(),
                            "event_id": f"{data_config.name.lower()}_noise_{hex_id}",
                            "trace_sampling_rate_hz": samplerate,
                            "trace_category": "noise",
                            "trace_start_time": starttime.isoformat()                        
                        }
                        
                    batch_rows.append(metadata_dict)
                    nsamples += 1            
                    if nsamples == config.batchsize:
                        dict2csv(batch_rows, header)
                        header = False
                        batch_rows.clear()
                        nsamples = 0


        metadata_grp.create_dataset(
            "bucket_ndp",
            data=np.array(buckets, dtype=np.int32)
        )

        
    if len(failed_reads) > 0:
        logger.warning(
            "Failed SEGY Reads:\n%s",
            "\n".join(map(str, failed_reads))
        )        
 
    if len(batch_rows) > 0:
        dict2csv(batch_rows, header)
    elif header:
        logger.warning(
            "Warning: Empty csv - no data extracted!"
        )
        return None
    
    # Assign Train-Test-Validation split
    split_df = pd.read_csv(csv_path, usecols=["split", "event_id"])
    split_df_multi = split_df[split_df["split"] == "multi"].copy()
    
    p_test = 1 - config.traintestsplit
    p_val = config.traintestsplit * (1-config.trainvalsplit)
    p_train = config.traintestsplit * config.trainvalsplit
       
    if config.eventsplit:
        event_ids = split_df_multi["event_id"].unique()
        event_splits = rng.choice(
            ["train", "dev", "test"],
            size=len(event_ids),
            p=[p_train, p_val, p_test]
        )

        event_split_map = dict(zip(event_ids, event_splits))
        split_df_multi["split"] = split_df_multi["event_id"].map(event_split_map)
        
    else:
        split_df_multi["split"] = rng.choice(
            ["train", "dev", "test"],
            size=len(split_df_multi),
            p=[p_train, p_val, p_test]
        )

    split_df.loc[split_df_multi.index, "split"] = split_df_multi["split"]    
    
    metadata_df = pd.read_csv(csv_path, chunksize=config.batchsize)
            
    with open(csv_path.parent / (csv_path.name + ".tmp"), "w") as temp:
        header = True
        start = 0
        for chunk in metadata_df:
            end = start + len(chunk)
            chunk["split"] = split_df["split"].iloc[start:end].values
            chunk.to_csv(temp, index=False, header=header)
            header = False
            start = end

    os.replace(csv_path.parent / (csv_path.name + ".tmp"), csv_path)

    return None