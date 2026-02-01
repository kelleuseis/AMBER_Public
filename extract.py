'''
Executable file for extracting Seisbench-compatible .hdf5 and .csv files 
from AMBER raw .segy files. Extraction parameters configured through 
extract_cfg.yaml.
'''

from dataloaders import MultiDataConfig, extract_data

from omegaconf import DictConfig
import hydra, logging



logging.basicConfig(
    level=logging.INFO,  # Default logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Standard format
)
logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path=".", config_name="extract_cfg")
def main(cfg: DictConfig):
    data_cfg = MultiDataConfig(**cfg['data_config'])

    try:
        logger.info("Starting data extraction...")
        extract_data(data_cfg)
    except Exception as e:
        logger.exception(f"Extraction failed due to an error: {e}")

    logger.info("Extraction complete.")
    
if __name__ == "__main__":
    main()