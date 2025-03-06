"""
Final iteration of LES calibration (or so we hope)

Kirby Heck
2025 January 24
"""

from WES2024.LES_new import compare_superposition


if __name__ == "__main__": 
    compare_superposition.run(
        regenerate=True,
        methods=["LESnew_nocontrol"],  # only calibrate to no control case
        fname="final_calibration_parameters.json"
    )
