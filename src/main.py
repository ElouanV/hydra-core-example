import hydra
import omegaconf
import logging
from pathlib import Path
import os

from datasets.dataset_loader import DatasetLoader
from models.model_selector import ModelSelector


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: omegaconf.DictConfig) -> None:

    logger = logging.getLogger(__name__)

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
    with open(Path("log/") / "config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)
        
    return score



if __name__ == "__main__":
    main()