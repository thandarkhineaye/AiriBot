import os

from omegaconf import OmegaConf, DictConfig


def load_config() -> DictConfig:
    """
    load config from base config + config based on env
    :return:
    """
    env_name = os.environ.get("ENV")
    if not env_name or env_name not in ["develop", "production", "staging"]:
        env_name = "develop"
    base_conf = OmegaConf.load(os.path.join(os.path.dirname(__file__), f'../config/config.yaml'))
    env_conf = OmegaConf.load(os.path.join(os.path.dirname(__file__), f'../config/config_{env_name}.yaml'))
    conf = OmegaConf.merge(base_conf, env_conf)
    return conf
