import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_histogram(values: npt.NDArray[np.float32]) -> plt.Figure:
    sns.set_theme(palette="flare")
    fig, ax = plt.subplots(1, 1)
    sns.histplot(
        pd.DataFrame({"value": values}),
        x="value",
        kde=True,
        ax=ax,
    )

    return fig
