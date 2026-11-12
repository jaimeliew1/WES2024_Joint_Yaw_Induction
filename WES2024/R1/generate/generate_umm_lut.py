# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: the two Unified Momentum Model look-up tables of WES2024.CustomRotors, written next to that module (git-ignored):
    unified_momentum_model_Ctprime_table.csv   UnifiedMomentumLUT (actuator-disk rotors, UnifiedLUTAD)
    unified_momentum_model_Ct_table.csv        ThrustBasedUnifiedMomentumLUT (BEM rotors, BEMUnifiedMomentumLUT)
Both are otherwise built lazily on first use; building them here first keeps the parallel workers of the later steps from
each building and writing them at the same time.

Usage: python -m WES2024.R1.generate.generate_umm_lut [--regenerate]
"""

import argparse

from WES2024.CustomRotors import CACHE_FN_CT, CACHE_FN_CTPRIME, ThrustBasedUnifiedMomentumLUT, UnifiedMomentumLUT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()
    for cls, fn in [(UnifiedMomentumLUT, CACHE_FN_CTPRIME), (ThrustBasedUnifiedMomentumLUT, CACHE_FN_CT)]:
        print(f"{cls.__name__}: {'regenerating' if args.regenerate or not fn.exists() else 'cached'} {fn}", flush=True)
        cls(regenerate=args.regenerate)
