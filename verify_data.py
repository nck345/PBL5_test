from src.utils import load_config
from src.dataset import prepare_data

config = load_config("configs/config.yaml")
loaders_info = prepare_data(config, verbose=True)
print("Data loaded successfully.")
