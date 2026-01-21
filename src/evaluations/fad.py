# from fadtk import FrechetAudioDistance, CLAPLaionModel

# class _CLAPSingleton:
#     _model: CLAPLaionModel | None = None

#     @classmethod
#     def get_model(cls) -> CLAPLaionModel:
#         if cls._model is None:
#             cls._model = CLAPLaionModel("music")
#         return cls._model
#
# this needs torch>=2.3.0, maybe some day

import os
import subprocess
from pathlib import Path


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


FADTK_LOCATION = "/home/jp/fadtk"


def fad(reference_set: str, target_set: str, fadtk_location: str = FADTK_LOCATION) -> float:
    ref_set = str(Path(reference_set).resolve())
    tgt_set = str(Path(target_set).resolve())
    with cd(fadtk_location):
        fadtk_run = subprocess.run(
            ["CUDA_VISIBLE_DEVICES=\"\"", "uv", "run", "python", "-m", "fadtk", "clap-laion-audio", ref_set, tgt_set, "--inf"],
            text=True,
            capture_output=True,
        )
        fad_score = float(fadtk_run.stdout.split()[-1])
    return fad_score
