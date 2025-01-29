import pandas as pd
import numpy as np
import matplotlib.axes as maxes
from typing import Iterable, Optional

def equal_or_bothna(a, b):
    return a == b or (pd.isna(a) and pd.isna(b))

def plot_series(ax: maxes.Axes, y: Iterable[Iterable[float]], group_labels: Optional[Iterable[str]] = None, bar_labels: Optional[Iterable[str]] = None, colors: Optional[list] = None):
    n_groups = len(y[0])
    n_series = len(y)
    bar_width = 1/(n_series+1)/1.2
    centers = np.arange(1, n_groups+1)
    offsets = np.linspace(-(n_series-1), (n_series-1), n_series)/(2*n_series)/1.2
    for i in range(n_series):
        if colors is not None:
            ax.bar(centers+offsets[i], y[i], width=bar_width, label=bar_labels[i] if bar_labels is not None else None, color=colors[i])
        else:
            ax.bar(centers+offsets[i], y[i], width=bar_width, label=bar_labels[i] if bar_labels is not None else None, color=colors[i])
    ax.set_xticks(centers, group_labels, rotation=30)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False