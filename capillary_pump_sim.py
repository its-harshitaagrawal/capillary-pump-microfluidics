#!/usr/bin/env python3
"""
Computational Design of a Self-Powered Microfluidic Pump Using Capillary Action
================================================================================
Python simulation accompanying the project proposal.

Models implemented
------------------
1. Lucas-Washburn (LW)      h(t) = sqrt( gamma r cos(theta) t / (2 mu) )        [ideal cylinder, horizontal]
2. Bosanquet / full ODE     rho d/dt(h h') = 2 gamma cos(theta)/r - 8 mu h h'/r^2 - rho g h sin(phi)
                            (adds inertia -> correct early-time behaviour, and gravity -> Jurin height)
3. Closed rectangular duct  LW-type law with exact Poiseuille resistance of a rectangular cross-section
4. Parametric design study  fill time vs (r, theta, fluid) with practical constraints + closed-form optimum

Run:  python capillary_pump_sim.py [output_dir]
Outputs: fig1..fig7 (PNG) and results.json in output_dir.

All quantities are SI unless stated. Edit the DESIGN CONSTRAINTS block to change the study.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 9.81  # m/s^2

# ----------------------------------------------------------------------------------------------
# Fluid library (representative values near 20-25 C; blood plasma values are approximate)
# ----------------------------------------------------------------------------------------------
FLUIDS = {
    "Water":              dict(gamma=0.072, mu=1.0e-3, rho=998.0),
    "Ethanol":            dict(gamma=0.022, mu=1.2e-3, rho=789.0),
    "Plasma (approx.)":   dict(gamma=0.055, mu=1.5e-3, rho=1025.0),
    "50% Glycerol-water": dict(gamma=0.068, mu=6.0e-3, rho=1126.0),
}

# ----------------------------------------------------------------------------------------------
# DESIGN CONSTRAINTS (assumed targets for a point-of-care style capillary pump -- edit freely)
# ----------------------------------------------------------------------------------------------
L_CHANNEL = 0.030      # channel length to be filled [m]
T_MAX = 20.0           # maximum acceptable fill time [s]
DP_MIN = 1000.0        # minimum capillary (pumping) pressure to push through downstream features [Pa]
R_MIN, R_MAX = 10e-6, 500e-6   # fabrication limit and footprint limit on effective radius [m]
BO_MAX = 0.1           # Bond number limit so gravity/orientation is negligible
THETA_MIN_DEG = 10.0   # best contact angle realistically reachable (plasma-treated glass/PDMS)
THETA_MAX_DEG = 80.0   # worst contact angle considered usable


# ==============================================================================================
# Core models
# ==============================================================================================
def _cos_pos(theta_deg):
    """cos(theta), clipped at 0: for theta >= 90 deg the liquid does not wick (no spontaneous flow)."""
    return np.maximum(np.cos(np.radians(theta_deg)), 0.0)


def washburn_h(t, gamma, r, theta_deg, mu):
    """Lucas-Washburn front position h(t) [m]. Returns 0 for theta >= 90 deg."""
    return np.sqrt(gamma * r * _cos_pos(theta_deg) * np.asarray(t) / (2.0 * mu))


def washburn_v(t, gamma, r, theta_deg, mu):
    """Front velocity dh/dt [m/s] (diverges as t -> 0 -- unphysical, see Bosanquet)."""
    t = np.asarray(t, dtype=float)
    return 0.5 * np.sqrt(gamma * r * _cos_pos(theta_deg) / (2.0 * mu * t))


def washburn_time_to_fill(L, gamma, r, theta_deg, mu):
    """Time for the front to reach distance L: t = 2 mu L^2 / (gamma r cos theta). Infinite for theta >= 90 deg."""
    c = _cos_pos(theta_deg)
    with np.errstate(divide="ignore"):
        out = np.where(c > 0, 2.0 * mu * L ** 2 / (gamma * r * np.where(c > 0, c, 1.0)), np.inf)
    return float(out) if np.ndim(out) == 0 else out


def capillary_pressure(gamma, r, theta_deg):
    """Young-Laplace capillary pressure for a cylinder [Pa]."""
    return 2.0 * gamma * np.cos(np.radians(theta_deg)) / r


def jurin_height(gamma, r, theta_deg, rho):
    """Equilibrium rise height in a vertical capillary [m]."""
    return 2.0 * gamma * np.cos(np.radians(theta_deg)) / (rho * G * r)


def bond_number(rho, r, gamma):
    return rho * G * r ** 2 / gamma


def bosanquet(t_end, gamma, r, theta_deg, mu, rho, g=0.0, sin_phi=1.0, h0=1e-7, n=800, t_start=1e-7):
    """
    Full 1-D capillary-rise ODE (Bosanquet 1923 with gravity):
        rho d/dt(h h') = P - 8 mu h h' / r^2 - rho g sin(phi) h,     P = 2 gamma cos(theta) / r
    Written with state (h, p = h h') to avoid the singularity at h = 0:
        h' = p / h ,    p' = (P - 8 mu p / r^2 - rho g sin(phi) h) / rho
    Returns t, h, v (v = p/h).
    """
    P = capillary_pressure(gamma, r, theta_deg)

    def rhs(t, y):
        h, p = y
        return [p / h, (P - 8.0 * mu * p / r ** 2 - rho * g * sin_phi * h) / rho]

    t_eval = np.logspace(np.log10(t_start), np.log10(t_end), n)
    t_eval[-1] = t_end  # guard against floating-point overshoot of the integration span
    t_eval = np.clip(t_eval, 0.0, t_end)
    sol = solve_ivp(rhs, (0.0, t_end), [h0, 0.0], method="Radau", t_eval=t_eval,
                    rtol=1e-9, atol=[1e-13, 1e-13])
    h, p = sol.y
    return sol.t, h, p / h


# ---- rectangular duct ------------------------------------------------------------------------
def rect_factor(aspect, exact=True):
    """
    Geometry factor F(a), a = depth/width <= 1, in the rectangular-duct hydraulic resistance
        R' = 12 mu / ( w d^3 F ).
    exact=True : full Fourier series (Bruus, Theoretical Microfluidics);  exact=False : 1 - 0.63 a.
    """
    a = np.atleast_1d(np.asarray(aspect, dtype=float))
    if not exact:
        out = 1.0 - 0.63 * a
    else:
        n = np.arange(1, 400, 2)[:, None]
        out = 1.0 - np.sum(192.0 / (n ** 5 * np.pi ** 5) * a * np.tanh(n * np.pi / (2.0 * a)), axis=0)
    return out if np.ndim(aspect) else out[0]


def rect_washburn_h(t, gamma, w, d, theta_deg, mu, exact=True):
    """
    Closed rectangular duct (width w, depth d, all four walls at the same contact angle):
        dP = 2 gamma cos(theta) (1/w + 1/d)
        h dh/dt = dP d^2 F / (12 mu)    ->   h = sqrt( gamma cos(theta) (1/w + 1/d) d^2 F t / (3 mu) )
    """
    if d > w:
        w, d = d, w
    F = rect_factor(d / w, exact)
    k = gamma * np.cos(np.radians(theta_deg)) * (1 / w + 1 / d) * d ** 2 * F / (3.0 * mu)
    return np.sqrt(k * np.asarray(t))


def hydraulic_radius(w, d):
    """
    Equivalent radius r_eq = 2A/P = D_h/2 = w d / (w + d)  (half the hydraulic diameter D_h = 4A/P).
    For a circle 2A/P equals the radius, so this is the radius of the 'equivalent' circular tube.
    (Note: textbooks call A/P the hydraulic radius; the function name is kept for compatibility.)
    """
    return w * d / (w + d)


# ==============================================================================================
# Verification
# ==============================================================================================
def run_verification():
    checks = []
    f = FLUIDS["Water"]
    r, th = 50e-6, 30.0
    tau = f["rho"] * r ** 2 / (8 * f["mu"])  # viscous relaxation time

    # 1. late-time agreement Bosanquet -> Washburn. The full model lags LW by a fixed start-up delay,
    #    so the relative gap should decay like tau/t: check (a) it is < 1 % for t > 200 tau and
    #    (b) err * t / tau is roughly constant (O(1)) across a decade of t.
    t, h, v = bosanquet(5.0, f["gamma"], r, th, f["mu"], f["rho"])
    hw = washburn_h(t, f["gamma"], r, th, f["mu"])
    mask = t > 200 * tau
    err = np.max(np.abs(h[mask] / hw[mask] - 1))
    sel = [(t > 100 * tau) & (t < 200 * tau), (t > 1000 * tau) & (t < 2000 * tau)]
    c1, c2 = [np.mean(np.abs(1 - h[m_] / hw[m_]) * t[m_] / tau) for m_ in sel]
    checks.append(dict(name="Late-time: full ODE -> Lucas-Washburn (error ~ tau/t)",
                       value=f"max rel. error {err:.2e} for t > 200 tau; err*t/tau = {c1:.2f} and {c2:.2f}",
                       passed=bool(err < 0.01 and abs(c1 / c2 - 1) < 0.15)))

    # 2. early-time linear scaling h ~ t and analytic prefactor sqrt(P/rho)
    P = capillary_pressure(f["gamma"], r, th)
    m = (t > 2e-6) & (t < 2e-5)
    slope = np.polyfit(np.log(t[m]), np.log(h[m]), 1)[0]
    pref = np.mean(h[m] / t[m]) / np.sqrt(P / f["rho"])
    checks.append(dict(name="Early-time: h ~ t (inertial regime)",
                       value=f"log-log slope {slope:.3f} (expect 1); prefactor ratio {pref:.3f}",
                       passed=bool(abs(slope - 1) < 0.05 and abs(pref - 1) < 0.05)))

    # 3. Washburn exponent
    m2 = t > 20 * tau
    slope2 = np.polyfit(np.log(t[m2]), np.log(h[m2]), 1)[0]
    checks.append(dict(name="Viscous regime: h ~ t^(1/2)",
                       value=f"log-log slope {slope2:.3f} (expect 0.5)", passed=bool(abs(slope2 - 0.5) < 0.02)))

    # 4. Jurin height from vertical rise
    r2 = 100e-6
    tj, hj, vj = bosanquet(60.0, f["gamma"], r2, 20.0, f["mu"], f["rho"], g=G, sin_phi=1.0)
    hJ = jurin_height(f["gamma"], r2, 20.0, f["rho"])
    errJ = abs(hj[-1] / hJ - 1)
    checks.append(dict(name="Vertical rise saturates at Jurin height",
                       value=f"numerical {hj[-1]*1e3:.2f} mm vs analytic {hJ*1e3:.2f} mm ({errJ:.1e})",
                       passed=bool(errJ < 0.01)))

    # 5. scaling law checks
    base = washburn_h(1.0, f["gamma"], r, th, f["mu"])
    s_r = washburn_h(1.0, f["gamma"], 4 * r, th, f["mu"]) / base
    s_mu = washburn_h(1.0, f["gamma"], r, th, 4 * f["mu"]) / base
    checks.append(dict(name="Scaling: 4x radius -> 2x h ; 4x viscosity -> 0.5x h",
                       value=f"{s_r:.3f}, {s_mu:.3f}", passed=bool(abs(s_r - 2) < 1e-9 and abs(s_mu - 0.5) < 1e-9)))

    # 6. rectangular duct -> parallel plates limit  (a -> 0, F -> 1)
    F_small = float(rect_factor(1e-3))
    checks.append(dict(name="Rect. duct: F -> 1 for shallow channels (a = 0.001)",
                       value=f"F = {F_small:.5f}", passed=bool(abs(F_small - 1) < 1e-3)))

    # 7. rectangular duct square limit F(1) = 0.4217 (known value)
    F_sq = float(rect_factor(1.0))
    checks.append(dict(name="Rect. duct: square-duct factor F(1)",
                       value=f"F = {F_sq:.4f} (literature 0.4217)", passed=bool(abs(F_sq - 0.4217) < 5e-4)))

    # 8. rect. duct with a=1 vs circle of same equivalent radius (sanity on magnitude)
    d = 100e-6
    hr = rect_washburn_h(1.0, f["gamma"], d, d, 0.0, f["mu"])
    hc = washburn_h(1.0, f["gamma"], hydraulic_radius(d, d), 0.0, f["mu"])
    checks.append(dict(name="Square duct vs circle of equal equivalent radius 2A/P (same order)",
                       value=f"h_rect/h_circ = {hr/hc:.3f}", passed=bool(0.8 < hr / hc < 1.2)))
    return checks


# ==============================================================================================
# Plot helpers
# ==============================================================================================
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "legend.fontsize": 8,
})
COLORS = ["#1f4e79", "#c0392b", "#16a085", "#d68910", "#7d3c98", "#2c3e50"]


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ==============================================================================================
# Figures
# ==============================================================================================
def fig1_baseline(outdir, results):
    f = FLUIDS["Water"]
    r, th = 50e-6, 30.0
    tau = f["rho"] * r ** 2 / (8 * f["mu"])
    t, h, v = bosanquet(10.0, f["gamma"], r, th, f["mu"], f["rho"])
    hw = washburn_h(t, f["gamma"], r, th, f["mu"])
    vw = washburn_v(t, f["gamma"], r, th, f["mu"])

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].plot(t, hw * 1e3, color=COLORS[0], lw=2.4, label="Lucas–Washburn")
    ax[0].plot(t, h * 1e3, color=COLORS[1], lw=1.4, ls="--", label="Full ODE (inertia + viscous)")
    ax[0].set_xlim(0, 10)
    ax[0].set_xlabel("time t [s]")
    ax[0].set_ylabel("front position h [mm]")
    ax[0].set_title("(a) Front position (linear axes)")
    ax[0].legend()

    ax[1].loglog(t, hw, color=COLORS[0], lw=2, label="Lucas–Washburn")
    ax[1].loglog(t, h, color=COLORS[1], lw=1.8, ls="--", label="Full ODE")
    ax[1].loglog(t, 1.2 * np.sqrt(capillary_pressure(f["gamma"], r, th) / f["rho"]) * t, color="grey", ls=":", lw=1,
                 label="slope 1 (inertial)")
    ax[1].loglog(t, 0.5 * washburn_h(t, f["gamma"], r, th, f["mu"]), color="grey", ls="-.", lw=1,
                 label="slope 1/2 (viscous)")
    ax[1].axvline(tau, color="k", lw=0.8, ls=":")
    ax[1].text(tau * 1.15, 3e-7, r"$\tau=\rho r^2/8\mu$", fontsize=8)
    ax[1].set_ylim(1e-7, 1e-1)
    ax[1].set_xlabel("time t [s]")
    ax[1].set_ylabel("front position h [m]")
    ax[1].set_title("(b) Log–log: two regimes")
    ax[1].legend(loc="upper left")

    ax[2].loglog(t, vw, color=COLORS[0], lw=2, label="Lucas–Washburn (diverges)")
    ax[2].loglog(t, v, color=COLORS[1], lw=1.8, ls="--", label="Full ODE")
    ax[2].set_xlabel("time t [s]")
    ax[2].set_ylabel("front velocity dh/dt [m/s]")
    ax[2].set_title("(c) Front velocity")
    ax[2].legend()
    fig.suptitle(f"Baseline: water, r = {r*1e6:.0f} µm, θ = {th:.0f}°, horizontal", y=1.03, fontsize=11)
    results["fig1"] = dict(tau_s=tau, v_max_ode=float(np.max(v)), v_at_1s_mm_s=float(np.interp(1, t, v) * 1e3),
                           h_at_1s_mm=float(np.interp(1, t, hw) * 1e3), h_at_10s_mm=float(hw[-1] * 1e3))
    return save(fig, outdir, "fig1_baseline.png")


def fig2_parametric(outdir, results):
    f = FLUIDS["Water"]
    t = np.linspace(1e-4, 10, 600)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    base = dict(gamma=f["gamma"], r=50e-6, theta_deg=30.0, mu=f["mu"])

    def sweep(a, key, values, labels, title, xlabel=None):
        for c, val, lab in zip(COLORS, values, labels):
            kw = dict(base)
            kw[key] = val
            a.plot(t, washburn_h(t, **kw) * 1e3, color=c, lw=1.8, label=lab)
        a.set_title(title)
        a.set_xlabel("time t [s]")
        a.set_ylabel("front position h [mm]")
        a.legend(title=xlabel, title_fontsize=8)

    sweep(ax[0, 0], "r", [10e-6, 25e-6, 50e-6, 100e-6, 200e-6], ["10", "25", "50", "100", "200"],
          "(a) Channel radius", "r [µm]")
    sweep(ax[0, 1], "theta_deg", [0, 30, 60, 75, 85], ["0°", "30°", "60°", "75°", "85°"],
          "(b) Contact angle", "θ")
    sweep(ax[1, 0], "mu", [0.5e-3, 1e-3, 3e-3, 6e-3, 20e-3], ["0.5", "1", "3", "6", "20"],
          "(c) Dynamic viscosity", "µ [mPa·s]")
    sweep(ax[1, 1], "gamma", [0.022, 0.040, 0.055, 0.072], ["22 (ethanol)", "40", "55", "72 (water)"],
          "(d) Surface tension", "γ [mN/m]")
    fig.suptitle("Parametric sweeps of the Lucas–Washburn front (baseline: water, r = 50 µm, θ = 30°; one parameter varied)",
                 y=0.995, fontsize=11)
    fig.tight_layout()

    # sensitivity table: local log-sensitivity d ln h / d ln x is analytic for LW
    results["fig2"] = dict(
        exponents=dict(gamma=0.5, r=0.5, mu=-0.5, t=0.5),
        h_at_10s_mm=float(washburn_h(10.0, **base) * 1e3),
        cos_sensitivity="d ln h / d theta = -0.5 tan(theta); at 60 deg = %.2f per rad" % (-0.5 * np.tan(np.radians(60))),
    )
    return save(fig, outdir, "fig2_parametric.png")


def design_maps(outdir, results):
    """Design study: (r, theta) maps of fill time with feasibility window for each fluid."""
    r_grid = np.logspace(np.log10(3e-6), np.log10(1.5e-3), 400)
    th_grid = np.linspace(0, 88, 300)
    R, TH = np.meshgrid(r_grid, th_grid)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    table = {}
    for a, (name, f) in zip(axes.ravel(), FLUIDS.items()):
        t_fill = washburn_time_to_fill(L_CHANNEL, f["gamma"], R, TH, f["mu"])
        dP = capillary_pressure(f["gamma"], R, TH)
        Bo = bond_number(f["rho"], R, f["gamma"])
        feasible = ((t_fill <= T_MAX) & (dP >= DP_MIN) & (R >= R_MIN) & (R <= R_MAX) &
                    (Bo <= BO_MAX) & (TH <= THETA_MAX_DEG))
        pc = a.pcolormesh(R * 1e6, TH, np.log10(t_fill), shading="auto", cmap="viridis_r",
                          vmin=-2, vmax=3)
        a.set_xscale("log")
        a.contourf(R * 1e6, TH, feasible.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
        a.contour(R * 1e6, TH, t_fill, levels=[T_MAX], colors="white", linewidths=1.6)
        a.contour(R * 1e6, TH, dP, levels=[DP_MIN], colors="orange", linewidths=1.6, linestyles="--")
        a.axvline(R_MIN * 1e6, color="red", lw=1)
        a.axvline(R_MAX * 1e6, color="red", lw=1)
        # analytic optimum for theta_min
        thm = THETA_MIN_DEG
        r_star = 2 * f["gamma"] * np.cos(np.radians(thm)) / DP_MIN
        t_star = washburn_time_to_fill(L_CHANNEL, f["gamma"], r_star, thm, f["mu"])
        r_use = min(r_star, R_MAX)
        a.plot(r_use * 1e6, thm, marker="*", ms=14, mfc="gold", mec="k", ls="none")
        a.set_title(f"{name}")
        a.set_xlabel("effective radius r [µm]")
        a.set_ylabel("contact angle θ [deg]")
        cb = fig.colorbar(pc, ax=a, pad=0.02)
        cb.set_label(r"log$_{10}$ fill time [s]")

        # closed-form optimum table for several achievable contact angles
        rows = []
        for thv in (10.0, 30.0, 60.0):
            rs = 2 * f["gamma"] * np.cos(np.radians(thv)) / DP_MIN
            ts = washburn_time_to_fill(L_CHANNEL, f["gamma"], rs, thv, f["mu"])
            ok = bool((R_MIN <= rs <= R_MAX) and ts <= T_MAX and bond_number(f["rho"], rs, f["gamma"]) <= BO_MAX)
            rows.append(dict(theta=thv, r_star_um=rs * 1e6, t_fill_s=ts,
                             dP_Pa=DP_MIN, mean_speed_mm_s=L_CHANNEL / ts * 1e3, feasible=ok,
                             Bo=bond_number(f["rho"], rs, f["gamma"])))
        # feasible-window extent at theta = THETA_MIN
        mask = feasible & (np.abs(TH - th_grid[np.argmin(np.abs(th_grid - 30))]) < 1e-9)
        table[name] = dict(optimum_rows=rows, feasible_fraction=float(feasible.mean()),
                           r_window_theta30_um=[float(R[mask].min() * 1e6), float(R[mask].max() * 1e6)] if mask.any() else None,
                           theta_limit_deg=float(np.degrees(np.arccos(np.sqrt(
                               2 * f["mu"] * L_CHANNEL ** 2 * DP_MIN / (2 * f["gamma"] ** 2 * T_MAX))))) if
                           (2 * f["mu"] * L_CHANNEL ** 2 * DP_MIN / (2 * f["gamma"] ** 2 * T_MAX)) < 1 else None)
    fig.suptitle(f"Design maps: fill time for L = {L_CHANNEL*1e3:.0f} mm.  Hatched = feasible "
                 f"(t ≤ {T_MAX:.0f} s, ΔP ≥ {DP_MIN:.0f} Pa, {R_MIN*1e6:.0f}–{R_MAX*1e6:.0f} µm, Bo ≤ {BO_MAX}, θ ≤ {THETA_MAX_DEG:.0f}°).  "
                 "White: t = t_max;  orange: ΔP = ΔP_min;  ★ optimum at θ = %.0f°" % THETA_MIN_DEG, y=0.995, fontsize=10)
    fig.tight_layout()
    results["design"] = dict(L_mm=L_CHANNEL * 1e3, T_MAX=T_MAX, DP_MIN=DP_MIN, R_MIN_um=R_MIN * 1e6, R_MAX_um=R_MAX * 1e6,
                             BO_MAX=BO_MAX, THETA_MIN=THETA_MIN_DEG, THETA_MAX=THETA_MAX_DEG, table=table)
    return save(fig, outdir, "fig3_design_maps.png")


def fig4_tradeoff(outdir, results):
    """Speed vs pumping-pressure trade-off (both set by r) for each fluid."""
    r = np.logspace(np.log10(R_MIN / 2), np.log10(R_MAX * 2), 300)
    th = 30.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, (name, f) in zip(COLORS, FLUIDS.items()):
        t_fill = washburn_time_to_fill(L_CHANNEL, f["gamma"], r, th, f["mu"])
        dP = capillary_pressure(f["gamma"], r, th)
        ax[0].loglog(r * 1e6, t_fill, color=c, lw=2, label=name)
        ax[1].loglog(dP, t_fill, color=c, lw=2, label=name)
    ax[0].axhline(T_MAX, color="k", ls="--", lw=1)
    ax[0].text(r[0] * 1e6 * 1.1, T_MAX * 1.2, "t_max", fontsize=8)
    ax[0].axvspan(r[0] * 1e6, R_MIN * 1e6, color="red", alpha=0.08)
    ax[0].axvspan(R_MAX * 1e6, r[-1] * 1e6, color="red", alpha=0.08)
    ax[0].set_xlabel("effective radius r [µm]")
    ax[0].set_ylabel(f"fill time for L = {L_CHANNEL*1e3:.0f} mm [s]")
    ax[0].set_title("(a) Larger channels fill faster …")
    ax[0].legend()
    ax[1].axhline(T_MAX, color="k", ls="--", lw=1)
    ax[1].axvline(DP_MIN, color="orange", ls="--", lw=1.4)
    ax[1].set_xlabel("capillary pressure ΔP [Pa]")
    ax[1].set_ylabel("fill time [s]")
    ax[1].set_title("(b) … but pump less hard: trade-off front (θ = 30°)")
    ax[1].fill_betweenx([1e-2, T_MAX], DP_MIN, 1e5, color="green", alpha=0.08)
    ax[1].set_ylim(1e-2, 1e3)
    ax[1].legend()
    fig.tight_layout()
    return save(fig, outdir, "fig4_tradeoff.png")


def fig5_gravity(outdir, results):
    f = FLUIDS["Water"]
    th = 20.0
    radii = [25e-6, 100e-6, 300e-6, 1000e-6]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    info = []
    for c, r in zip(COLORS, radii):
        t, h, v = bosanquet(30.0, f["gamma"], r, th, f["mu"], f["rho"], g=G, sin_phi=1.0)
        hJ = jurin_height(f["gamma"], r, th, f["rho"])
        tt, hh, vv = bosanquet(30.0, f["gamma"], r, th, f["mu"], f["rho"], g=0.0)
        ax[0].semilogx(t, h / hJ, color=c, lw=1.8, label=f"r = {r*1e6:.0f} µm  (h$_J$ = {hJ*1e3:.1f} mm)")
        ax[1].semilogx(tt, hh * 1e3, color=c, lw=1.8, label=f"horizontal, r = {r*1e6:.0f} µm")
        info.append(dict(r_um=r * 1e6, hJ_mm=hJ * 1e3, Bo=bond_number(f["rho"], r, f["gamma"]),
                         overshoot=float(np.max(h) / hJ - 1)))
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_xlabel("time t [s]")
    ax[0].set_ylabel("h / h$_{Jurin}$")
    ax[0].set_title("(a) Vertical rise: gravity-limited (full ODE)")
    ax[0].legend(loc="upper left")
    ax[1].set_xlabel("time t [s]")
    ax[1].set_ylabel("front position h [mm]")
    ax[1].set_title("(b) Horizontal: no gravity limit (full ODE)")
    ax[1].legend(loc="upper left")
    fig.suptitle("Gravity and orientation effects (water, θ = 20°)", y=1.02, fontsize=11)
    fig.tight_layout()
    results["gravity"] = info
    return save(fig, outdir, "fig5_gravity.png")


def fig6_rect(outdir, results):
    f = FLUIDS["Water"]
    a = np.linspace(0.01, 1.0, 300)
    Fe, Fa = rect_factor(a, True), rect_factor(a, False)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    ax[0].plot(a, Fe, color=COLORS[0], lw=2, label="exact (Fourier series)")
    ax[0].plot(a, Fa, color=COLORS[1], lw=1.6, ls="--", label="1 − 0.63 a")
    ax[0].set_xlabel("aspect ratio a = d / w")
    ax[0].set_ylabel("resistance factor F")
    ax[0].set_title("(a) Rectangular-duct factor")
    ax[0].legend()

    # ratio of LW coefficient to a circle of equal equivalent radius (2A/P)
    w = 100e-6
    ratio = [rect_washburn_h(1.0, f["gamma"], w, ai * w, 0.0, f["mu"]) /
             washburn_h(1.0, f["gamma"], hydraulic_radius(w, ai * w), 0.0, f["mu"]) for ai in a]
    ax[1].plot(a, ratio, color=COLORS[2], lw=2)
    ax[1].axhline(1, color="k", ls=":", lw=1)
    ax[1].set_xlabel("aspect ratio a = d / w")
    ax[1].set_ylabel(r"h$_{rect}$ / h$_{LW}$(r = r$_{eq}$)")
    ax[1].set_title("(b) Error of equivalent-radius LW (r$_{eq}$ = 2A/P)")
    results["rect"] = dict(
        ratio_square=float(ratio[-1]), ratio_a0_1=float(np.interp(0.1, a, ratio)),
        F_square_exact=float(Fe[-1]), F_square_approx=float(Fa[-1]),
        max_approx_err_pct_a_le_0_5=float(np.max(np.abs(Fa[a <= 0.5] / Fe[a <= 0.5] - 1)) * 100))

    t = np.linspace(1e-4, 10, 500)
    d = 50e-6
    for c, wi in zip(COLORS, [50e-6, 100e-6, 200e-6, 500e-6]):
        ax[2].plot(t, rect_washburn_h(t, f["gamma"], wi, d, 30.0, f["mu"]) * 1e3, color=c, lw=1.8,
                   label=f"w = {wi*1e6:.0f} µm (a = {d/wi:.2f})")
    ax[2].plot(t, washburn_h(t, f["gamma"], hydraulic_radius(100e-6, d), 30.0, f["mu"]) * 1e3, color="grey", ls=":",
               label="LW, r = r$_{eq}$ (w = 100 µm)")
    ax[2].set_xlabel("time t [s]")
    ax[2].set_ylabel("front position h [mm]")
    ax[2].set_title("(c) Rectangular channel, d = 50 µm, θ = 30°")
    ax[2].legend()
    fig.tight_layout()
    return save(fig, outdir, "fig6_rectangular.png")


def fig7_validity(outdir, results):
    """When does inertia matter? Ratio full-ODE / LW, and crossover times vs radius (water, theta = 30)."""
    f = FLUIDS["Water"]
    th = 30.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    radii = [5e-6, 20e-6, 50e-6, 100e-6, 300e-6]
    cross = []
    for c, r in zip(COLORS, radii):
        t, h, v = bosanquet(5.0, f["gamma"], r, th, f["mu"], f["rho"])
        ratio = h / washburn_h(t, f["gamma"], r, th, f["mu"])
        ax[0].semilogx(t, ratio, color=c, lw=1.8, label=f"r = {r*1e6:.0f} µm")
    ax[0].axhline(1, color="k", ls=":", lw=1)
    ax[0].axhline(0.95, color="grey", ls="--", lw=0.8)
    ax[0].set_ylim(0, 1.15)
    ax[0].set_xlabel("time t [s]")
    ax[0].set_ylabel("h$_{full}$ / h$_{LW}$")
    ax[0].set_title("(a) Washburn over-predicts at early times")
    ax[0].legend()

    rs = np.logspace(np.log10(3e-6), np.log10(500e-6), 14)
    for name, fl, c in zip(["Water", "50% Glycerol-water"], [FLUIDS["Water"], FLUIDS["50% Glycerol-water"]],
                           [COLORS[0], COLORS[3]]):
        tc, tau_list = [], []
        for r in rs:
            tau = fl["rho"] * r ** 2 / (8 * fl["mu"])
            t, h, v = bosanquet(max(200 * tau, 1e-3), fl["gamma"], r, th, fl["mu"], fl["rho"], n=500)
            ratio = h / washburn_h(t, fl["gamma"], r, th, fl["mu"])
            idx = np.argmax(ratio >= 0.95)
            tc.append(t[idx] if ratio[idx] >= 0.95 else np.nan)
            tau_list.append(tau)
        ax[1].loglog(rs * 1e6, tc, "o-", color=c, ms=4, label=f"{name}: t$_{{95\\%}}$ (full ODE)")
        ax[1].loglog(rs * 1e6, tau_list, color=c, ls=":", label=f"{name}: τ = ρr²/8µ")
        if name == "Water":
            results["validity"] = dict(r_um=[float(x * 1e6) for x in rs], t95_water_s=[float(x) for x in tc],
                                       tau_water_s=[float(x) for x in tau_list])
    ax[1].set_xlabel("radius r [µm]")
    ax[1].set_ylabel("time [s]")
    ax[1].set_title("(b) Time for LW to be within 5 % of full model")
    ax[1].legend()
    fig.tight_layout()
    return save(fig, outdir, "fig7_validity.png")


# ==============================================================================================
def main(outdir="capillary_output"):
    os.makedirs(outdir, exist_ok=True)
    results = {}
    results["verification"] = run_verification()
    figs = [fig1_baseline(outdir, results), fig2_parametric(outdir, results), design_maps(outdir, results),
            fig4_tradeoff(outdir, results), fig5_gravity(outdir, results), fig6_rect(outdir, results),
            fig7_validity(outdir, results)]
    results["figures"] = figs
    results["fluids"] = FLUIDS
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("Verification checks:")
    for c in results["verification"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['value']}")
    print("Figures written to", outdir)
    return results


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Capillary micro-pump simulation (all values in SI units, angles in degrees).")
    ap.add_argument("outdir", nargs="?", default="capillary_output", help="output folder for figures and results.json")
    ap.add_argument("--L", type=float, default=L_CHANNEL, help="channel length to fill [m]")
    ap.add_argument("--tmax", type=float, default=T_MAX, help="maximum acceptable fill time [s]")
    ap.add_argument("--dpmin", type=float, default=DP_MIN, help="minimum capillary pressure [Pa]")
    ap.add_argument("--rmin", type=float, default=R_MIN, help="smallest radius (fabrication limit) [m]")
    ap.add_argument("--rmax", type=float, default=R_MAX, help="largest radius (footprint limit) [m]")
    ap.add_argument("--bomax", type=float, default=BO_MAX, help="maximum Bond number")
    ap.add_argument("--theta-min", type=float, default=THETA_MIN_DEG, help="best achievable contact angle [deg]")
    ap.add_argument("--theta-max", type=float, default=THETA_MAX_DEG, help="worst usable contact angle [deg]")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    L_CHANNEL, T_MAX, DP_MIN = a.L, a.tmax, a.dpmin
    R_MIN, R_MAX, BO_MAX = a.rmin, a.rmax, a.bomax
    THETA_MIN_DEG, THETA_MAX_DEG = a.theta_min, a.theta_max
    main(a.outdir)
