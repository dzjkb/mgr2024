import torch
from laion_clap import CLAP_Module

SAMPLING_RATE = 48000  # fixed for CLAP


class _CLAPSingleton:
    _model: CLAP_Module | None = None

    @classmethod
    def get_model(cls) -> CLAP_Module:
        if cls._model is None:
            cls._model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            cls._model.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")
        return cls._model


def get_embeddings(audio: torch.Tensor) -> torch.Tensor:
    """
    note: works on mono only, `audio` should be of `(n, t)` shape
    """

    assert len(audio.shape) == 2, f"clap accepts only mono audio, got a tensor with {len(audio.shape)} dimensions: {tuple(audio.shape)}"
    model = _CLAPSingleton.get_model()
    return model.get_audio_embedding_from_data(x=audio, use_tensor=False)
