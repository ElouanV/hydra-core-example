import hydra
import omegaconf
import logging
from pathlib import Path
import os

from datasets.dataset_loader import DatasetLoader
from models.model_selector import ModelSelector

def get_logger(config):
    """ Build a logger object to log the application

    :param config: Configuration object
    :type config: omegaconf.DictConfig
    :return: Logger object
    :rtype: logging.Logger
    """
    os.makedirs(config.log_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(Path(config.log_path) / "app.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    if config.console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.addHandler(file_handler)
    
    return logger

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    logger = get_logger(config)

    logger.info(f"Config: {omegaconf.OmegaConf.to_yaml(config)}")    
    config.models.param = config.models.param[config.datasets.name]

    try: 
        X_train, X_test, y_train, y_test = DatasetLoader(config).load()
        model = ModelSelector(config).get_model()
    except AssertionError as e:
        logger.error(e)
        return
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"Model score: {score}")

    # Save the configuration used to run the model
    with open(Path(config.log_path) / "config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    # We return the score for Optuna to optimize
    return score


if __name__ == "__main__":
    main()