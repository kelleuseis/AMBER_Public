'''
Executable file for extracting Seisbench-compatible .hdf5 and .csv files 
from AMBER raw .segy files. Extraction parameters configured through 
extract_cfg.yaml.
'''

from amber.database import MultiDataConfig, BackendConfig, extract_data

from omegaconf import DictConfig
import hydra, logging


logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path=".", config_name="extract_cfg")
def main(cfg: DictConfig):
    data_cfg = MultiDataConfig(**cfg['data_config'])
    be_cfg = BackendConfig(**cfg['backend_config'])

    try:
        logger.info("Starting data extraction...")
        extract_data(data_cfg, be_cfg)
    except Exception as e:
        logger.exception(f"Extraction failed due to an error: {e}")

    logger.info("Extraction complete.")
    
if __name__ == "__main__":
    main()