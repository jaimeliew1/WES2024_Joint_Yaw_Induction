import matplotlib.pyplot as plt
import polars as pl
import shutil

# Use Latex Fonts in all plots made in this module.
if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


pl.Config.set_tbl_rows(50)
