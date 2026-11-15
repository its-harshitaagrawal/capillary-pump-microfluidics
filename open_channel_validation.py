#!/usr/bin/env python3
"""
Validation against the proposal's reference paper
=================================================
Kolliopoulos, Jochem, Johnson, Suszynski, Francis & Kumar (2021), "Capillary-flow dynamics in open
rectangular microchannels", J. Fluid Mech. 911, A32, doi:10.1017/jfm.2020.986.

What this module implements (all equation numbers refer to the paper)
  * Modified Lucas-Washburn (MLW) model, eqs. (2)-(3):  z_m = sqrt(k t),
        k = 2 sigma H zeta_o(lambda) [cos(theta0)(1 + 2 lambda) - 1] / (3 mu),   lambda = H/W
        zeta_o(lambda) = 24/(pi^4 lambda^2) sum_n (2n+1)^-4 [1 - tanh((2n+1) pi lambda)/((2n+1) pi lambda)]
  * Critical aspect ratio, eq. (1):  lambda_c = (1 - sin theta0) / (2 cos theta0)
  * Dimensionless mobility, eq. (17b):  k* = k / (U L) = zeta_o [cos(theta0)(1 + 2 lambda) - 1] / 3
  * The paper's experimental setup (Section 2, Table 1): H = 22.5 um, L = 30 mm, W = 17-200 um,
    four liquids with measured properties and equilibrium contact angles.

What it does NOT implement: the paper's self-similar lubrication-theory-based (LTB) model, which needs a
finite-element velocity solver and a nonlinear boundary-value solver. The paper's measured meniscus curves
exist only as plotted figures, so they are NOT bundled here; instead the script accepts an optional CSV of
digitised experimental points (see --exp) and reports the agreement.

Run:  python open_channel_validation.py [output_dir] [--exp experimental.csv]
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import capillary_pump_sim as sim
from capillary_pump_sim import COLORS, rect_factor, rect_washburn_h, save

# ----------------------------------------------------------------------------------------------
# Paper's experimental setup (Section 2, Table 1)
# ----------------------------------------------------------------------------------------------
H_PAPER = 22.5e-6                                   # channel height [m]
L_PAPER = 30e-3                                     # channel length [m]
WIDTHS_UM = [17, 25, 50, 75, 100, 200]              # channel widths [um]
T_CAMERA_MAX = 400.0                                # s, maximum recording time quoted in the paper
LIQUIDS = {                                         # Table 1; mu in Pa.s, sigma in N/m
    "NOA74":            dict(rho=1000.0, mu=0.5411, sigma=29.1e-3, theta0=10.0),   # rho only quoted as ">1 g/cm3"
    "Silicone oil":     dict(rho=1070.0, mu=0.0430, sigma=23.5e-3, theta0=18.0),
    "Mineral oil":      dict(rho=838.0,  mu=0.0272, sigma=29.8e-3, theta0=32.0),
    "Propylene glycol": dict(rho=1040.0, mu=0.0434, sigma=33.8e-3, theta0=42.0),
}
# Paper's stated lambda_c values (caption of Fig. 9) used as a check of eq. (1)
LAMBDA_C_PAPER = {"NOA74": 0.42, "Silicone oil": 0.36, "Mineral oil": 0.28, "Propylene glycol": 0.22}

# Paper's qualitative statements about MLW vs experiment (Section 4.2), used when --exp is supplied.
#   +1: MLW under-predicts (k_exp/k_MLW > 1),  -1: MLW over-predicts (k_exp/k_MLW < 1),  0: "good agreement"
PAPER_CLAIMS = {
    ("NOA74", "lambda>=lambda_c"): 0, ("NOA74", "lambda<lambda_c"): -1,
    ("Silicone oil", "lambda>=lambda_c"): 0, ("Silicone oil", "lambda<lambda_c"): -1,
    ("Mineral oil", "lambda>=lambda_c"): +1, ("Mineral oil", "lambda<lambda_c"): -1,
    ("Propylene glycol", "lambda>=lambda_c"): +1,
}


# ==============================================================================================
# Model functions
# ==============================================================================================
def zeta_o(lam, nterms=400):
    """Aspect-ratio function, eq. (3). Accepts scalars or arrays."""
    scalar = np.ndim(lam) == 0
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    nterms = int(min(max(nterms, 5.0 / lam.min()), 200000))  # small lambda needs many terms (slow tail)
    n = (2 * np.arange(nterms) + 1)[:, None]
    x = n * np.pi * lam[None, :]
    out = 24.0 / (np.pi ** 4 * lam ** 2) * np.sum(n ** -4.0 * (1.0 - np.tanh(x) / x), axis=0)
    return out[0] if scalar else out


def k_dimensionless(lam, theta0_deg):
    """k* = zeta_o [cos(theta0)(1+2 lambda) - 1] / 3  (eq. 17b combined with eq. 2)."""
    return zeta_o(lam) * (np.cos(np.radians(theta0_deg)) * (1 + 2 * np.asarray(lam)) - 1.0) / 3.0


def k_mlw(sigma, mu, H, W, theta0_deg):
    """Dimensional mobility parameter k [m^2/s], eq. (2). Zero or negative means the MLW model predicts no flow."""
    lam = H / W
    return 2 * sigma * H * zeta_o(lam) * (np.cos(np.radians(theta0_deg)) * (1 + 2 * lam) - 1.0) / (3 * mu)


def z_mlw(t, k):
    return np.sqrt(np.maximum(k, 0.0) * np.asarray(t))


def lambda_c(theta0_deg):
    th = np.radians(theta0_deg)
    return (1 - np.sin(th)) / (2 * np.cos(th))


def lambda_noflow(theta0_deg):
    """Below this aspect ratio the MLW driving term cos(theta0)(1+2 lambda) - 1 is <= 0 (no predicted flow)."""
    th = np.radians(theta0_deg)
    return (1 - np.cos(th)) / (2 * np.cos(th))


def optimum_aspect_ratio(theta0_deg):
    """lambda maximising k* (fastest filling for fixed H, liquid and theta0). Fine grid + local refinement."""
    lam = np.logspace(-2, 1.5, 4000)
    k = k_dimensionless(lam, theta0_deg)
    i = int(np.argmax(k))
    lam2 = np.linspace(lam[max(i - 2, 0)], lam[min(i + 2, len(lam) - 1)], 2000)
    k2 = k_dimensionless(lam2, theta0_deg)
    j = int(np.argmax(k2))
    return float(lam2[j]), float(k2[j])


# ==============================================================================================
# Verification of the implementation against independent / published results
# ==============================================================================================
def run_checks():
    checks = []

    # V1: zeta_o equals an independent expression built from the closed-duct Fourier series used in the main
    #     simulation. An open channel with a shear-free flat top is half of a closed duct of height 2H, so
    #     zeta_o = M m^3 F(m/M) / (8 H^3 W) with m = min(2H, W), M = max(2H, W).
    lams = np.logspace(-1.5, 1.5, 40)
    H = 1.0
    rel = []
    for lam in lams:
        W = H / lam
        m, M = min(2 * H, W), max(2 * H, W)
        z_ind = M * m ** 3 * rect_factor(m / M, True) / (8 * H ** 3 * W)
        rel.append(abs(zeta_o(lam) / z_ind - 1))
    checks.append(dict(name="zeta_o (paper eq. 3) vs independent half-duct Fourier-series identity",
                       value=f"max rel. difference {max(rel):.1e} over lambda = 0.03-30", passed=bool(max(rel) < 1e-6)))

    # V2: limits. zeta_o -> 1 - 1.2604 lambda (= F(2 lambda), F ~ 1 - 0.6302 a) as lambda -> 0
    #     and zeta_o -> 1/(4 lambda^2) as lambda -> infinity.
    lam0 = 1e-3
    z0 = float(zeta_o(lam0))
    asym0 = 1 - 2 * 0.63021 * lam0
    zinf = float(zeta_o(1e3) * 1e3 ** 2)
    checks.append(dict(name="zeta_o limits: -> 1 - 1.260 lambda as lambda -> 0; -> 1/(4 lambda^2) as lambda -> infinity",
                       value=f"zeta_o(1e-3) = {z0:.6f} vs {asym0:.6f};  lambda^2 zeta_o(1e3) = {zinf:.5f} (expect 0.25)",
                       passed=bool(abs(z0 - asym0) < 1e-5 and abs(zinf - 0.25) < 1e-3)))

    # V3: lambda_c values quoted in Fig. 9 caption
    got = {k: round(float(lambda_c(v["theta0"])), 2) for k, v in LIQUIDS.items()}
    ok = all(abs(got[k] - LAMBDA_C_PAPER[k]) < 0.006 for k in got)
    checks.append(dict(name="lambda_c from eq. (1) reproduces the paper's quoted values (0.42, 0.36, 0.28, 0.22)",
                       value=", ".join(f"{k}: {got[k]:.2f}" for k in got), passed=bool(ok)))

    # V4: theta0 = 0, deep narrow channel -> MLW tends to the closed parallel-plate Lucas-Washburn law with gap W,
    #     which is also the narrow-dimension limit of the closed-duct model in the main simulation.
    lam = 100.0
    W = H_PAPER / lam
    sg, mu = 0.03, 0.05
    kk = float(k_mlw(sg, mu, H_PAPER, W, 0.0))
    k_plate = sg * W / (3 * mu)
    k_closed = float(rect_washburn_h(1.0, sg, W, H_PAPER, 0.0, mu)) ** 2
    checks.append(dict(name="theta0 = 0, lambda = 100: MLW -> parallel-plate law sigma W/(3 mu) and closed-duct model",
                       value=f"k_MLW/k_plate = {kk/k_plate:.4f};  k_MLW/k_closed-duct = {kk/k_closed:.4f}",
                       passed=bool(abs(kk / k_plate - 1) < 0.02 and abs(kk / k_closed - 1) < 0.02)))

    # V5: consistency of the experimental design with statements in the paper
    lam_set = {W: H_PAPER / (W * 1e-6) for W in WIDTHS_UM}
    msgs = []
    ok5 = True
    # the paper's text quotes lambda = 0.45, 0.225 (NOA74), 0.3 and 0.23 (mineral oil) and lambda_c(NOA74)=0.42
    for target, Wum in ((0.45, 50), (0.225, 100), (0.30, 75), (0.23, 100)):
        d = abs(lam_set[Wum] - target)
        ok5 &= d < 0.006
        msgs.append(f"W={Wum}: {lam_set[Wum]:.3f}")
    both = {k: (any(lam_set[W] >= lambda_c(v["theta0"]) for W in WIDTHS_UM),
                any(lam_set[W] < lambda_c(v["theta0"]) for W in WIDTHS_UM)) for k, v in LIQUIDS.items()}
    # propylene glycol: only W = 200 um lies below lambda_c; the paper ran only lambda > lambda_c for it
    pg_below = [W for W in WIDTHS_UM if lam_set[W] < lambda_c(LIQUIDS["Propylene glycol"]["theta0"])]
    ok5 &= (pg_below == [200]) and all(both[k][0] and both[k][1] for k in ("NOA74", "Silicone oil", "Mineral oil"))
    checks.append(dict(name="Channel widths give the aspect ratios quoted in the paper (0.45, 0.225, 0.3, 0.23)",
                       value="; ".join(msgs) + f";  PG widths below lambda_c: {pg_below} um", passed=bool(ok5)))

    # V6: k* positivity threshold consistent with lambda_noflow
    th = 42.0
    lamn = float(lambda_noflow(th))
    s1, s2 = float(k_dimensionless(lamn * 0.98, th)), float(k_dimensionless(lamn * 1.02, th))
    checks.append(dict(name="MLW driving term changes sign at lambda = (1 - cos theta0)/(2 cos theta0)",
                       value=f"theta0 = 42 deg: threshold {lamn:.3f}; k* = {s1:.2e} below, {s2:.2e} above",
                       passed=bool(s1 < 0 < s2)))
    return checks


# ==============================================================================================
# Predictions in the paper's geometry
# ==============================================================================================
def predictions():
    rows = []
    for name, p in LIQUIDS.items():
        lc = float(lambda_c(p["theta0"]))
        for Wum in WIDTHS_UM:
            W = Wum * 1e-6
            lam = H_PAPER / W
            k = float(k_mlw(p["sigma"], p["mu"], H_PAPER, W, p["theta0"]))
            flows = k > 0
            t_fill = L_PAPER ** 2 / k if flows else None
            rows.append(dict(liquid=name, W_um=Wum, lam=lam, regime="lambda>=lambda_c" if lam >= lc else "lambda<lambda_c",
                             lam_c=lc, k_um2_per_s=k * 1e12 if flows else None, k_star=float(k_dimensionless(lam, p["theta0"])),
                             z_at_100s_mm=float(z_mlw(100.0, k) * 1e3) if flows else 0.0,
                             t_to_30mm_s=t_fill, reaches_end_within_400s=bool(flows and t_fill <= T_CAMERA_MAX)))
    return rows


# ==============================================================================================
# Optional comparison with digitised experimental data
# ==============================================================================================
def compare_with_experiment(csv_path, tol=0.25):
    """
    CSV columns: liquid, W_um, t_s, z_mm  (digitised from the paper's Fig. 9; liquid names as in LIQUIDS).
    For every (liquid, W) curve the experimental mobility k_exp = mean(z^2 / t) is compared with MLW.
    'tol' is our tolerance for calling agreement 'good' (+-25 %); the paper gives no numeric criterion.
    """
    data = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(r for r in fh if not r.lstrip().startswith("#")):
            data.setdefault((row["liquid"].strip(), float(row["W_um"])), []).append((float(row["t_s"]), float(row["z_mm"]) * 1e-3))
    out = []
    for (liq, Wum), pts in sorted(data.items()):
        p = LIQUIDS[liq]
        t = np.array([a for a, _ in pts])
        z = np.array([b for _, b in pts])
        keep = (t > 0) & (z > 0)          # z^2/t is undefined at t = 0
        t, z = t[keep], z[keep]
        k_exp = float(np.mean(z ** 2 / t))
        k_th = float(k_mlw(p["sigma"], p["mu"], H_PAPER, Wum * 1e-6, p["theta0"]))
        lam = H_PAPER / (Wum * 1e-6)
        regime = "lambda>=lambda_c" if lam >= lambda_c(p["theta0"]) else "lambda<lambda_c"
        ratio = k_exp / k_th if k_th > 0 else float("nan")
        claim = PAPER_CLAIMS.get((liq, regime))
        if claim is None or not np.isfinite(ratio):
            verdict = "no paper statement"
        elif claim == 0:
            verdict = "consistent" if abs(ratio - 1) <= tol else "NOT consistent"
        elif claim > 0:
            verdict = "consistent" if ratio > 1 else "NOT consistent"
        else:
            verdict = "consistent" if ratio < 1 else "NOT consistent"
        # RMS relative error of z(t) predicted by MLW
        zt = z_mlw(t, k_th)
        rms = float(np.sqrt(np.mean((zt / z - 1) ** 2))) if k_th > 0 else float("nan")
        out.append(dict(liquid=liq, W_um=Wum, lam=lam, regime=regime, k_exp_um2_s=k_exp * 1e12, k_mlw_um2_s=k_th * 1e12,
                        k_ratio=ratio, rms_rel_err_z=rms, paper_claim={1: "MLW under-predicts", -1: "MLW over-predicts", 0: "good agreement"}.get(claim, "-"),
                        verdict=verdict, n_points=len(pts)))
    return out


# ==============================================================================================
# Figures
# ==============================================================================================
def val_fig1(outdir, preds):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    lam = np.logspace(-1.2, 1.0, 400)
    for c, (name, p) in zip(COLORS, LIQUIDS.items()):
        ks = k_dimensionless(lam, p["theta0"])
        ax[0].semilogx(lam, np.where(ks > 0, ks, np.nan), color=c, lw=2, label=f"{name} (θ₀ = {p['theta0']:.0f}°)")
        ax[0].axvline(lambda_c(p["theta0"]), color=c, ls=":", lw=1)
        lam_exp = np.array([H_PAPER / (w * 1e-6) for w in WIDTHS_UM])
        ke = k_dimensionless(lam_exp, p["theta0"])
        ax[0].plot(lam_exp[ke > 0], ke[ke > 0], "o", color=c, ms=5, mfc="white")
    ax[0].set_xlabel("aspect ratio λ = H / W")
    ax[0].set_ylabel("dimensionless mobility k* = k / (U L)")
    ax[0].set_title("(a) MLW mobility vs λ  (dotted: λ$_c$; circles: widths used in the paper)")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(0.06, 10)

    for c, (name, p) in zip(COLORS, LIQUIDS.items()):
        k = np.array([k_mlw(p["sigma"], p["mu"], H_PAPER, w * 1e-6, p["theta0"]) for w in WIDTHS_UM]) * 1e12
        ax[1].loglog(WIDTHS_UM, np.where(k > 0, k, np.nan), "o-", color=c, lw=1.8, label=name)
    ax[1].set_xlabel("channel width W [µm]  (H = 22.5 µm)")
    ax[1].set_ylabel("mobility k [µm²/s]")
    ax[1].set_title("(b) Dimensional mobility in the paper's geometry")
    ax[1].legend()
    fig.tight_layout()
    return save(fig, outdir, "val_fig1_mobility.png")


def val_fig2(outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    t = np.linspace(0.01, T_CAMERA_MAX, 800)
    for a, (name, p) in zip(axes.ravel(), LIQUIDS.items()):
        lc = lambda_c(p["theta0"])
        for c, Wum in zip(COLORS, WIDTHS_UM):
            k = float(k_mlw(p["sigma"], p["mu"], H_PAPER, Wum * 1e-6, p["theta0"]))
            lam = H_PAPER / (Wum * 1e-6)
            if k <= 0:
                continue
            a.plot(t, z_mlw(t, k) * 1e3, color=c, lw=1.8, ls="-" if lam >= lc else "--",
                   label=f"W = {Wum} µm (λ = {lam:.2f})")
        a.axhline(L_PAPER * 1e3, color="k", lw=0.8, ls=":")
        a.set_ylim(0, 33)
        a.set_title(f"{name}:  θ₀ = {p['theta0']:.0f}°, λ$_c$ = {lc:.2f}")
        a.set_xlabel("time t [s]")
        a.set_ylabel("meniscus position z$_m$ [mm]")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("MLW predictions in the paper's geometry (solid: λ ≥ λ$_c$; dashed: λ < λ$_c$; dotted: 30 mm channel end). "
                 "No experimental data overlaid.", y=0.995, fontsize=10)
    fig.tight_layout()
    return save(fig, outdir, "val_fig2_mlw_curves.png")


def val_fig3(outdir, res):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
    lam = np.logspace(-1.2, 1.2, 500)
    for c, th in zip(COLORS, (0, 10, 18, 32, 42)):
        ks = k_dimensionless(lam, th)
        ax[0].semilogx(lam, np.where(ks > 0, ks, np.nan), color=c, lw=1.8, label=f"θ₀ = {th}°")
        lo, ko = optimum_aspect_ratio(th)
        ax[0].plot(lo, ko, "*", ms=11, color=c, mec="k")
    ax[0].set_xlabel("aspect ratio λ = H / W")
    ax[0].set_ylabel("k*")
    ax[0].set_title("(a) A fastest aspect ratio exists (★)")
    ax[0].legend()

    ths = np.linspace(0, 55, 40)
    lopt = [optimum_aspect_ratio(t)[0] for t in ths]
    ax[1].plot(ths, lopt, color=COLORS[0], lw=2, label="λ$_{opt}$ (max k*)")
    ax[1].plot(ths, lambda_c(ths), color=COLORS[1], lw=1.6, ls="--", label="λ$_c$ (eq. 1)")
    ax[1].plot(ths, lambda_noflow(ths), color="grey", lw=1.4, ls=":", label="MLW no-flow limit")
    ax[1].set_xlabel("equilibrium contact angle θ₀ [deg]")
    ax[1].set_ylabel("aspect ratio λ")
    ax[1].set_title("(b) Optimum aspect ratio vs θ₀")
    ax[1].legend(loc="upper left")

    # open vs closed duct
    for c, th in zip(COLORS, (10, 32)):
        lam2 = np.logspace(-0.7, 1.0, 120)
        ratio = []
        for l in lam2:
            W = H_PAPER / l
            ko = float(k_mlw(0.03, 0.05, H_PAPER, W, th))
            kc = float(rect_washburn_h(1.0, 0.03, W, H_PAPER, th, 0.05)) ** 2
            ratio.append(ko / kc if ko > 0 else np.nan)
        ax[2].semilogx(lam2, ratio, color=c, lw=2, label=f"θ₀ = {th}°")
    ax[2].axhline(1, color="k", ls=":", lw=1)
    ax[2].set_xlabel("aspect ratio λ = H / W")
    ax[2].set_ylabel("k$_{open\\ (MLW)}$ / k$_{closed\\ duct}$")
    ax[2].set_title("(c) Why the closed-duct model cannot stand in")
    ax[2].legend()
    fig.tight_layout()
    return save(fig, outdir, "val_fig3_optimum.png")


# ==============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="validation_output")
    ap.add_argument("--exp", default=None, help="CSV of digitised experimental points (liquid,W_um,t_s,z_mm)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    res = {"checks": run_checks(), "predictions": predictions()}
    opt = {}
    for name, p in LIQUIDS.items():
        lo, ko = optimum_aspect_ratio(p["theta0"])
        opt[name] = dict(theta0=p["theta0"], lam_opt=lo, k_star_max=ko, W_opt_um=H_PAPER / lo * 1e6,
                         lam_c=float(lambda_c(p["theta0"])), lam_noflow=float(lambda_noflow(p["theta0"])))
        # k = k* * U L = k* * 2 H sigma / mu
        opt[name]["k_max_um2_s"] = ko * 2 * H_PAPER * p["sigma"] / p["mu"] * 1e12
    res["optimum"] = opt
    res["figures"] = [val_fig1(args.outdir, res["predictions"]), val_fig2(args.outdir), val_fig3(args.outdir, res)]
    if args.exp:
        res["experimental_comparison"] = compare_with_experiment(args.exp)
    with open(os.path.join(args.outdir, "validation_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("Implementation checks against the paper and independent identities:")
    for c in res["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['value']}")
    print("\nOptimum aspect ratio (MLW, H = 22.5 um):")
    for n, o in opt.items():
        print(f"  {n:17s} theta0={o['theta0']:.0f}  lambda_opt={o['lam_opt']:.2f}  (W_opt={o['W_opt_um']:.0f} um)  k*_max={o['k_star_max']:.4f}")
    if args.exp:
        print("\nComparison with supplied experimental data:")
        for r in res["experimental_comparison"]:
            print(f"  {r['liquid']:17s} W={r['W_um']:.0f}um {r['regime']:17s} k_exp/k_MLW={r['k_ratio']:.2f}  "
                  f"paper: {r['paper_claim']:20s} -> {r['verdict']}")
    print("\nOutputs in", args.outdir)
    return res


if __name__ == "__main__":
    main()
