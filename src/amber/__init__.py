import logging, os, requests
from pathlib import Path
from tqdm import tqdm

AMBER_DATA_ROOT = Path(
    os.getenv("AMBER_DATA_ROOT", Path.home() / ".amber")
)
AMBER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AMBER_DATA_ROOT", str(AMBER_DATA_ROOT))

AMBER_DATA_RAW = AMBER_DATA_ROOT / 'raw'
AMBER_DATA_COMPILED = AMBER_DATA_ROOT / 'compiled'


DATASETS = {
    "amber_default": {
        "filename": "amber_raw_segys.zip",
        "url": "https://zenodo.org/records/18944111/files/amber_raw_segys.zip"
    },
}


logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][PID:%(process)d][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    
    
def _download(src, dest):
    '''
    Parameters
    ------------------------------------------
    src: str or pathlib.Path
    
    dest: str or pathlib.Path
    '''
    logger.info(f"Downloading data from {src} to {dest.name}...")

    response = requests.get(src, stream=True, timeout=60)
    response.raise_for_status()

    dest_temp = dest.with_suffix(".tmp")

    with open(dest_temp, "wb") as f, tqdm(
        total=int(response.headers.get("content-length", 0)),
        unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name,
    ) as pbar:

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    dest_temp.replace(dest)

    logger.info("Download complete.")
    
    
def ensure_dataset(datasetname="amber_default", autodownload=True):
    datasetname = datasetname.lower()

    if datasetname not in DATASETS:
        msg = f"Unknown dataset '{datasetname}'. Available: {list(DATASETS.keys())}"
        logger.error(msg)
        raise ValueError(msg)

    AMBER_DATA_RAW.mkdir(parents=True, exist_ok=True)
    datasetinfo = DATASETS[datasetname]
    datasetpath = AMBER_DATA_RAW / datasetinfo["filename"]

    if datasetpath.exists():
        return datasetpath

    msg = f"{datasetinfo['filename']} not found in {AMBER_DATA_RAW}"
    if not autodownload:
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(msg)
    logger.info("Starting download...")

    _download(datasetinfo["url"], datasetpath)

    return datasetpath


__all__ = [
    "AMBER_DATA_ROOT",
    "AMBER_DATA_RAW",
    "AMBER_DATA_COMPILED",
    ensure_dataset,
]