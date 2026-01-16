import numpy as np
import matplotlib.pyplot as plt
from tests import test_BEM_models_gch, test_BEM_models_original
from WES2024 import utils

df_original = test_BEM_models_original.generate()
df_gch = test_BEM_models_gch.generate()

def plot():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for pitch in df_original['pitch'].unique()[:3]:
        _df_orig = df_original.filter(df_original['pitch'] == pitch).sort('tsr')
        _df_gch = df_gch.filter(df_gch['pitch'] == pitch).sort('tsr')

        ax[0].plot(_df_orig['tsr'], _df_orig['Ct'], label=f'Original Model {pitch:.1f}°', linestyle='-')
        ax[0].plot(_df_gch['tsr'], _df_gch['Ct'], label=f'GCH Model {pitch:.1f}°', linestyle='--')

        ax[1].plot(_df_orig['tsr'], _df_orig['Cp'], label=f'Original Model {pitch:.1f}°', linestyle='-')
        ax[1].plot(_df_gch['tsr'], _df_gch['Cp'], label=f'GCH Model {pitch:.1f}°', linestyle='--')

    ax[0].set_xlabel('TSR')
    ax[0].set_ylabel('Ct')
    ax[0].set_title('Ct vs TSR')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_xlabel('TSR')
    ax[1].set_ylabel('Cp')
    ax[1].set_title('Cp vs TSR')
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(utils.FIGDIR / 'BEM_models_comparison.png')

if __name__ == "__main__":
    plot()

