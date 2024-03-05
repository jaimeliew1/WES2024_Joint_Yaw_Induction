import matplotlib.pyplot as plt
import polars as pl


# Use Latex Fonts in all plots made in this module.
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


pl.Config.set_tbl_rows(50)
