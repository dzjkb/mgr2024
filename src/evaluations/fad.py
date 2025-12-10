import torch
from fadtk import FrechetAudioDistance, CLAPLaionModel

class _CLAPSingleton:
    _model: CLAPLaionModel | None = None

    @classmethod
    def get_model(cls) -> CLAPLaionModel:
        if cls._model is None:
            cls._model = CLAPLaionModel("music")
        return cls._model


def fad(baseline: torch.Tensor, eval_set: torch.Tensor) -> float:
    return 0.0