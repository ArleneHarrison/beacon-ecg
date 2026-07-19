#!/usr/bin/env python3
"""Sample size for a prospective validation of BEACON-ECG.

Effect sizes are this study's own measured values, not assumptions:
  HR 1.85 (discordance), AUROC 0.900/0.860/0.789, OR 1.66 (subgroup miscalibration).
"""
import math
Z, Zb = 1.959964, 0.8416          # alpha 0.05 two-sided, power 0.80

def events_for_hr(hr, p_high=0.2):
    """Schoenfeld: events needed to detect a hazard ratio."""
    return (Z + Zb)**2 / (p_high * (1 - p_high) * math.log(hr)**2)

def se_auc(A, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUROC."""
    Q1 = A / (2 - A); Q2 = 2*A*A / (1 + A)
    return math.sqrt((A*(1-A) + (n_pos-1)*(Q1-A*A) + (n_neg-1)*(Q2-A*A)) / (n_pos*n_neg))

if __name__ == "__main__":
    d = events_for_hr(1.85)
    print(f"Prognostic endpoint: {d:.0f} deaths needed to detect HR 1.85")
    for m in (0.02, 0.03, 0.05, 0.08):
        print(f"  at {m*100:.0f}% 1-year mortality -> follow {d/m:,.0f} patients")
    print("\nDiagnostic endpoints (target: >=100 events and 95% CI half-width <=0.05):")
    for name, A, prev in [("HFrEF", 0.90, 0.10), ("severe AS", 0.86, 0.02), ("LVH", 0.79, 0.06)]:
        for N in (1000, 2000, 3000, 5000):
            n1 = max(int(N*prev), 1); half = 1.96*se_auc(A, n1, N-n1)
            if half <= 0.05 and n1 >= 100:
                print(f"  {name:<10} prevalence {prev*100:>3.0f}%  ->  N={N:,} ({n1} events, ±{half:.3f})")
                break
        else:
            print(f"  {name:<10} prevalence {prev*100:>3.0f}%  ->  >5,000 needed")
