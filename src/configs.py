from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def parse_hydra_config(root_configs_dir: str, path: str) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    abs_training_dir_path = Path(root_configs_dir).resolve()

    assert Path(path).exists(), f"experiment config {path} does not exist"
    relative_path = path.removeprefix("configs/experiment/")

    with initialize_config_dir(str(abs_training_dir_path), version_base=None):
        config = compose(config_name="default", overrides=[f"+experiment={relative_path}"])

    return cast(dict[str, Any], OmegaConf.to_object(config))
