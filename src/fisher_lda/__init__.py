from .lda import FisherLDA
from .data import load_dry_bean
from . import assumptions, plots, benchmark, benchmark_plots

__all__ = ["FisherLDA", "load_dry_bean", "assumptions", "plots", "benchmark", "benchmark_plots"]
__version__ = "0.3.0"
