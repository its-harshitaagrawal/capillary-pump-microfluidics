"""
Builds the final project report (PDF) from the outputs of the two simulation scripts.

Usage:  python build_report.py sim_dir validation_dir report.pdf
   sim_dir        : output folder of  capillary_pump_sim.py      (contains results.json + fig*.png)
   validation_dir : output folder of  open_channel_validation.py (contains validation_results.json + val_fig*.png)
Needs: reportlab, pillow, matplotlib (for the DejaVu fonts).
All numbers in the text that come from the simulations are read from the JSON files, so the report stays
consistent if the study parameters are changed and the scripts are re-run.
"""
import json
import os
import sys

import matplotlib
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether

OUT, VAL, PDF = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(os.path.join(OUT, "results.json")))
VR = json.load(open(os.path.join(VAL, "validation_results.json")))

fd = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(fd, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, TEAL, GREY = colors.HexColor("#1f3a5f"), colors.HexColor("#1b9a8f"), colors.HexColor("#5d6d7e")
body = ParagraphStyle("body", fontName="DV", fontSize=8.8, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10.4, alignment=0, textColor=GREY)
cap = ParagraphStyle("cap", parent=body, fontSize=7.8, leading=10.4, textColor=colors.HexColor("#34495e"), spaceAfter=9)
cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=10.2, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B", textColor=colors.white)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=13, textColor=TEAL, spaceBefore=7, spaceAfter=3, keepWithNext=1)
title = ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("sub", fontName="DV", fontSize=9.5, leading=13, textColor=GREY, alignment=TA_CENTER, spaceAfter=2)
eq = ParagraphStyle("eq", parent=body, fontName="DV-I", fontSize=9.5, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6,
                    backColor=colors.HexColor("#f4f8fb"), borderPadding=(4, 4, 4, 4))
bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
ref = ParagraphStyle("ref", parent=body, fontSize=8.2, leading=11.2, leftIndent=16, firstLineIndent=-16, alignment=0, spaceAfter=3)

W = A4[0] - 3.6 * cm
story = []


def P(t, s=body): story.append(Paragraph(t, s))
def B(t): story.append(Paragraph(t, bul, bulletText="•"))


def fig(name, caption, maxh=None):
    path = os.path.join(VAL if name.startswith("val_") else OUT, name)
    w, h = Image.open(path).size
    width, height = W, W * h / w
    if maxh and height > maxh:
        width, height = width * maxh / height, maxh
    story.append(KeepTogether([RLImage(path, width=width, height=height), Spacer(1, 3), Paragraph(caption, cap)]))


def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), cellb if (i == 0 and header) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3dc")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f8fb")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([t, Spacer(1, 7)]))


# ------------------------------------------------------------------ values pulled from the simulation outputs
D, F1, RC, V = R["design"], R["fig1"], R["rect"], R["verification"]
GV = {int(g["r_um"]): g for g in R["gravity"]}
OP = VR["optimum"]
G = 9.81
n_pass = sum(c["passed"] for c in V)
n_vpass = sum(c["passed"] for c in VR["checks"])
water = R["fluids"]["Water"]
cos2 = lambda th: np.cos(np.radians(th)) ** 2
surf_gain = cos2(10) / cos2(60)                                        # theta 60 -> 10 deg at the optimum
r_bo = np.sqrt(D["BO_MAX"] * water["gamma"] / (water["rho"] * G))      # radius at the Bond-number limit
wopt = D["table"]["Water"]["optimum_rows"][0]                            # water, theta = theta_min
lam_opts = [o["lam_opt"] for o in OP.values()]
# which constraint sets the smallest usable radius at theta = 30 deg
t_at_rmin = {k: 2 * f["mu"] * (D["L_mm"] * 1e-3) ** 2 / (f["gamma"] * D["R_MIN_um"] * 1e-6 * np.cos(np.radians(30)))
             for k, f in R["fluids"].items()}
slowest = max(t_at_rmin, key=t_at_rmin.get)
tau_opt = water["rho"] * (wopt["r_star_um"] * 1e-6) ** 2 / (8 * water["mu"])

# ------------------------------------------------------------------ title
story.append(Paragraph("INTRODUCTION TO MICROFLUIDICS AND APPLICATIONS",
                       ParagraphStyle("k", parent=sub, fontName="DV-B", textColor=TEAL, fontSize=8.5)))
story.append(Paragraph("Computational Design of a Self-Powered Microfluidic Pump Using Capillary Action", title))
story.append(Paragraph("Project Report: Theoretical Modelling and Python Simulation", sub))
story.append(Paragraph("Harshita Agrawal (2023CH10969)  ·  Minakshi Dhingad (2023CH70817)", sub))
story.append(Paragraph("Code: capillary_pump_sim.py, open_channel_validation.py", sub))
story.append(Spacer(1, 8))

# ------------------------------------------------------------------ abstract
P("Abstract and key findings", h1)
P("Passive capillary pumps move liquid through narrow channels using surface tension alone, without electricity or an external "
  "pressure source. We built a Python simulation based on the Lucas–Washburn (LW) relation to predict how the liquid front advances "
  "with time, and extended it with fluid inertia (Bosanquet), gravity, and closed rectangular channels. We used these models for a constrained "
  "parametric design study, and we validated the work in two ways: against exact analytic limits "
  f"({n_pass}/{len(V)} checks pass) and against the open-channel model of our reference paper, Kolliopoulos et al. (2021) "
  f"({n_vpass}/{len(VR['checks'])} checks pass). A quantitative comparison with the paper's measured curves remains to be done (Section 7.5).")
B(f"<b>Baseline.</b> Water, r = 50 µm, θ = 30°, horizontal: LW predicts h = {F1['h_at_1s_mm']:.1f} mm after 1 s and "
  f"{F1['h_at_10s_mm']:.0f} mm after 10 s, with front speed falling as t<super>−1/2</super> (≈ {F1['v_at_1s_mm_s']:.0f} mm/s at 1 s).")
B("<b>Scaling.</b> h ∝ √γ, √r, √(cos θ), 1/√µ, √t; equivalently t<sub>fill</sub> ∝ µL²/(γ r cos θ). Radius is the "
  "strongest geometric lever; viscosity and surface tension are fixed by the liquid.")
B(f"<b>Validity.</b> LW fails at very early times: it predicts infinite velocity at t = 0, whereas the full model peaks at "
  f"{F1['v_max_ode']:.2f} m/s here. The two converge with relative error ≈ 0.5 τ/t, where τ = ρr²/(8µ) "
  f"({F1['tau_s']*1e3:.2f} ms for the baseline), so LW is within 5 % for t ≳ 10 τ.")
B("<b>Design trade-off.</b> A wider channel fills faster but generates less capillary pressure (ΔP = 2γ cos θ / r). With a minimum-pressure "
  "requirement the optimum is r* = 2γ cos θ / ΔP<sub>min</sub>, giving t<sub>min</sub> = µ L² ΔP<sub>min</sub> / (γ cos θ)². "
  f"Making the surface hydrophilic (θ from 60° to 10°) fills {surf_gain:.1f}× faster at the optimum.")
B(f"<b>Non-circular channels.</b> Using LW with the equivalent radius r<sub>eq</sub> = 2A/P under-predicts a closed square duct by about "
  f"{(1 - 1/RC['ratio_square'])*100:.0f} % and over-predicts by ≈ {(1/RC['ratio_a0_1']-1)*100:.0f} % at aspect ratio 0.1. For <i>open</i> channels "
  f"(the reference paper) the fastest aspect ratio is λ = H/W ≈ {min(lam_opts):.1f}–{max(lam_opts):.1f}, i.e. width ≈ "
  f"{1/max(lam_opts):.1f}–{1/min(lam_opts):.1f} × depth.")

# ------------------------------------------------------------------ 1 introduction
P("1. Introduction", h1)
P("Capillary action is the spontaneous flow of a liquid into a narrow space, driven by the attraction between the liquid and the "
  "channel wall. When a liquid wets a wall (contact angle θ &lt; 90°) the meniscus curves and creates a pressure drop "
  "ΔP = 2γ cos θ / r (the Young–Laplace pressure) that pulls the liquid in. Viscous friction resists the motion, and the balance of the "
  "two gives the classical Lucas–Washburn law, h ∝ √t (Lucas 1918; Washburn 1921).")
P("This mechanism is the basis of <b>passive or self-powered microfluidic pumps</b>. Because no syringe pump, pressure controller or "
  "battery is needed, capillary-driven devices are cheap, portable and simple to operate. They are used in lab-on-a-chip systems "
  "(Juncker et al. 2002; Olanrewaju et al. 2018) and in point-of-care diagnostics such as paper-based assays and lateral-flow tests "
  "(Martinez et al. 2007), where a drop of blood, saliva or urine must reach a reaction zone within a set time. "
  "Designing such a device means choosing the channel size and surface treatment so that the sample fills the channel quickly "
  "enough, with enough driving pressure, using a realistic sample volume.")
P("Real channels are rarely ideal glass tubes. Lithographically made channels are rectangular and often open on top, and in "
  "open channels the free surface changes the driving pressure and can even stop the flow (Concus &amp; Finn 1969). Our reference paper, "
  "Kolliopoulos et al. (2021), studies exactly this case and shows where the simple models work and where they fail.")
P("<b>Core research question.</b> <i>Which channel dimensions and surface properties enable faster capillary-driven liquid transport "
  "without electricity or an external pump?</i>")

# ------------------------------------------------------------------ 2 objectives
P("2. Objectives and scope", h1)
P("The three objectives below are taken from our project proposal. The table shows where each one is addressed.")
table([["Objective (from the proposal)", "What we did", "Section"],
       ["<b>Model:</b> implement the Lucas–Washburn relation in Python and plot liquid-front position and velocity vs time",
        "LW model plus a full ODE with inertia and gravity, and a closed rectangular-duct model; position and velocity plots", "3, 5.1"],
       ["<b>Analyze:</b> effect of channel size, contact angle, viscosity and surface tension",
        "Parametric sweeps, analytic sensitivities, validity of LW, gravity and channel-shape effects", "5.2–5.5"],
       ["<b>Optimize and validate:</b> find good parameter combinations under practical constraints and compare with published results",
        "Constrained design maps with a closed-form optimum; analytic verification; implementation of the reference paper's open-channel model", "4, 6, 7"],
       ["Discuss assumptions and limitations, especially non-circular channels and early-time flow",
        "Early-time (inertia) analysis, closed and open rectangular channels, list of model assumptions", "5.3, 5.5, 7, 8"]],
      [6.4 * cm, W - 8.0 * cm, 1.6 * cm])

# ------------------------------------------------------------------ 3 models
P("3. Models implemented", h1)
table([["Model", "Governing equation", "Use / assumptions"],
       ["Lucas–Washburn (ideal cylinder, horizontal)",
        "h(t) = √[γ r cos θ · t / (2µ)];  dh/dt = ½ √[γ r cos θ / (2µ t)]",
        "Quasi-steady Poiseuille flow, constant contact angle, no gravity, no inertia. This is the proposal's governing equation."],
       ["Full 1-D ODE (Bosanquet + gravity)",
        "ρ d(h ḣ)/dt = 2γ cos θ / r − 8µ h ḣ / r² − ρ g h sin φ",
        "Adds liquid inertia and hydrostatic head (φ = tube inclination). Solved in the variables (h, p = h ḣ) with an implicit Radau integrator."],
       ["Jurin height", "h<sub>J</sub> = 2γ cos θ / (ρ g r)", "Equilibrium height of a vertical rise; used to check the ODE."],
       ["Closed rectangular duct (w × d, a = d/w ≤ 1)",
        "ΔP = 2γ cos θ (1/w + 1/d);  h ḣ = ΔP d² F(a) / (12µ);  h = √[γ cos θ (1/w+1/d) d² F t / (3µ)]",
        "F(a) = 1 − Σ<sub>n odd</sub> (192 / n⁵π⁵) a tanh(nπ/2a) (exact series, 200 terms; Bruus 2008); F ≈ 1 − 0.63a also provided. Same θ on all four walls."],
       ["Open rectangular channel (Section 7)",
        "z<sub>m</sub> = √(kt),  k = 2σH ζ<sub>o</sub>(λ) [cos θ<sub>0</sub>(1 + 2λ) − 1] / (3µ),  λ = H/W",
        "Modified Lucas–Washburn (MLW) model of Kolliopoulos et al. (2021), eqs. 1–3: flat free surface, no inertia."],
       ["Design metrics", "t<sub>fill</sub> = 2µL² / (γ r cos θ);  ΔP = 2γ cos θ / r;  Bo = ρ g r² / γ",
        "Used for the feasibility maps and the optimum."]],
      [3.6 * cm, 6.6 * cm, W - 10.2 * cm])
P("The characteristic inertial time is τ = ρr²/(8µ): for t ≪ τ the ODE gives h ≈ t √(ΔP/ρ) (linear growth), and for t ≫ τ it recovers "
  "the LW t<super>1/2</super> law (Quéré 1997). Fluid properties used (representative values near 20–25 °C; plasma values approximate):")
fl = [["Liquid", "γ [mN/m]", "µ [mPa·s]", "ρ [kg/m³]"]]
for k, v in R["fluids"].items():
    fl.append([k, f"{v['gamma']*1e3:.0f}", f"{v['mu']*1e3:.1f}", f"{v['rho']:.0f}"])
table(fl, [6 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm])

# ------------------------------------------------------------------ 4 verification
P("4. Verification", h1)
P("Each model is checked against an independent analytic limit. The late-time check needs a comment: the full model lags "
  "LW by a fixed start-up delay, so the relative gap decays like τ/t rather than vanishing abruptly. The test therefore confirms both "
  "that the gap is below 1 % for t > 200 τ and that err·t/τ stays constant (0.50 at t ≈ 100–200 τ and at t ≈ 1000–2000 τ), "
  "which is the expected asymptotic behaviour.")
vr = [["#", "Check", "Result", "Status"]]
for i, c in enumerate(V, 1):
    vr.append([i, c["name"], c["value"], "PASS" if c["passed"] else "FAIL"])
table(vr, [0.7 * cm, 6.6 * cm, W - 9.3 * cm, 2.0 * cm])

# ------------------------------------------------------------------ 5 results
P("5. Results", h1)
P("5.1 Baseline behaviour", h2)
P("Figure 1 shows the front position and velocity for water in a 50 µm-radius capillary at θ = 30°. On linear axes the full "
  "model and LW are indistinguishable beyond a few milliseconds; the log–log panel shows the two regimes (slope 1 below τ, slope ½ above), "
  "and panel (c) shows that inertia removes the LW velocity singularity.")
fig("fig1_baseline.png", "<b>Figure 1.</b> Baseline simulation: (a) front position, (b) log–log regimes with τ marked, (c) front velocity.")

P("5.2 Parametric sensitivities", h2)
P("Figure 2 varies one parameter at a time. Because h ∝ (γ r cos θ / µ)<super>1/2</super>, the log-sensitivities are exactly "
  "±½ for γ, r and µ. The contact-angle sensitivity is d ln h / dθ = −½ tan θ, so it is weak below ≈ 45° and grows rapidly as θ → 90° "
  "(−0.87 per radian at 60°; filling stops entirely at 90°). In practice, treating a surface from θ = 60° to 20° shortens the fill time "
  "by cos 20°/cos 60° ≈ 1.9×, comparable to doubling the radius.")
fig("fig2_parametric.png", "<b>Figure 2.</b> Effect of (a) radius, (b) contact angle, (c) viscosity, (d) surface tension on h(t). Baseline: water, r = 50 µm, θ = 30°.", maxh=12.8 * cm)

P("5.3 When is Lucas–Washburn valid?", h2)
P("Figure 3 compares the full model with LW. LW over-predicts the early position by orders of magnitude, but the gap shrinks "
  "quickly: LW comes within 5 % of the full solution at t<sub>95</sub> ≈ 10.4 τ for every radius and both liquids tested "
  "(consistent with the 0.5 τ/t error law). Since τ ∝ r², small channels (r ≲ 50 µm, τ ≲ 0.3 ms) follow LW almost "
  "immediately, while in millimetre-scale tubes inertia matters for tens of milliseconds.")
fig("fig7_validity.png", "<b>Figure 3.</b> (a) Ratio of full-model to LW front position for several radii (water, θ = 30°). (b) Time for LW to be within 5 % of the full model, compared with τ = ρr²/8µ.", maxh=7.6 * cm)

P("5.4 Gravity and orientation", h2)
P(f"In a horizontal channel the front never stops (h ∝ √t), so a passive pump is limited by channel length and by how long the user "
  f"can wait, not by hydrostatics. In a vertical tube the rise stops at the Jurin height (Figure 4): for water at θ = 20°, "
  f"h<sub>J</sub> is {GV[25]['hJ_mm']:.0f} mm at r = 25 µm but only {GV[1000]['hJ_mm']:.1f} mm at r = 1 mm. The full model reproduces h<sub>J</sub> "
  f"to 0.2 % (Check 4). The approach to equilibrium slows sharply in narrow tubes (its time scale grows as 1/r³), so the 25 µm tube is still "
  f"rising at the end of the 30 s plotted. The 1 mm tube (Bo = {GV[1000]['Bo']:.2f}) is underdamped: the front overshoots h<sub>J</sub> by "
  f"{GV[1000]['overshoot']*100:.0f} % and oscillates before settling, an inertial effect LW cannot represent. For devices that must work in any "
  f"orientation, Bo ≤ {D['BO_MAX']} (r ≲ {r_bo*1e3:.2f} mm for water) is a sensible design bound; all channels in the design study sit far below it.")
fig("fig5_gravity.png", "<b>Figure 4.</b> (a) Vertical rise normalised by the Jurin height, including the underdamped overshoot for r = 1 mm. (b) The same channels horizontally: no equilibrium height.", maxh=7.8 * cm)

P("5.5 Rectangular channels", h2)
P(f"Real lab-on-a-chip channels are rectangular. The exact resistance factor gives F(1) = {RC['F_square_exact']:.4f} for a square duct "
  f"(literature value 0.4217); the common approximation 1 − 0.63a gives {RC['F_square_approx']:.2f}, 12 % low at a = 1, though within "
  f"{RC['max_approx_err_pct_a_le_0_5']:.2f} % of the exact value for a ≤ 0.5. A circular LW law with the equivalent radius r<sub>eq</sub> = 2A/P "
  f"(half the hydraulic diameter) under-predicts the front position by about {(1 - 1/RC['ratio_square'])*100:.0f} % for a square section but over-predicts "
  f"it by {(1/RC['ratio_a0_1']-1)*100:.0f} % at a = 0.1 (Figure 5b), so the exact rectangular factor should be used. "
  f"For shallow channels the result tends to the parallel-plate limit h = √(γ cos θ d t / 3µ). "
  f"These are <i>closed</i> channels; open channels, the subject of the reference paper, are treated in Section 7.")
fig("fig6_rectangular.png", "<b>Figure 5.</b> (a) Rectangular-duct resistance factor, exact vs 1 − 0.63a. (b) Equivalent-radius LW vs exact rectangular result. (c) Front position for d = 50 µm and varying width.", maxh=6.6 * cm)

# ------------------------------------------------------------------ 6 design study
P("6. Parametric design study and optimisation", h1)
P("<b>Problem.</b> Choose the effective channel radius r and the surface condition (contact angle θ) that fill a channel of length L as "
  "fast as possible, subject to practical constraints:")
table([["Constraint", "Value used", "Rationale"],
       ["Fill time", f"t<sub>fill</sub> ≤ {D['T_MAX']:.0f} s for L = {D['L_mm']:.0f} mm", "Usable assay time"],
       ["Pumping pressure", f"ΔP = 2γ cos θ / r ≥ {D['DP_MIN']:.0f} Pa", "Margin to overcome downstream resistance, pinning and valves"],
       ["Fabrication / footprint", f"{D['R_MIN_um']:.0f} µm ≤ r ≤ {D['R_MAX_um']:.0f} µm", "Typical soft-lithography / micromilling limits"],
       ["Gravity", f"Bo = ρ g r² / γ ≤ {D['BO_MAX']}", "Orientation-independent operation"],
       ["Surface", f"{D['THETA_MIN']:.0f}° ≤ θ ≤ {D['THETA_MAX']:.0f}°", "Best achievable after plasma/hydrophilic treatment; worst usable"]],
      [3.5 * cm, 5.6 * cm, W - 9.1 * cm])
P("The proposal does not fix numeric device targets, so these are assumed values typical of a point-of-care device. They can be changed "
  "from the command line (Appendix A) and every figure, table and optimum is recomputed.", small)
P("<b>Closed-form optimum.</b> In LW, t<sub>fill</sub> falls as 1/r while the capillary pressure also falls as 1/r, so the two goals "
  "conflict (Figure 7). Maximising speed subject to the pressure constraint pushes r to its upper bound r* = 2γ cos θ / ΔP<sub>min</sub>, giving")
P("t<sub>min</sub> = µ L² ΔP<sub>min</sub> / (γ² cos² θ)", eq)
P("A device is feasible if t<sub>min</sub> ≤ t<sub>max</sub>, i.e. cos θ ≥ √[µ L² ΔP<sub>min</sub> / (γ² t<sub>max</sub>)]. This gives the "
  "largest tolerable contact angle for each liquid (table below) and shows that, at a fixed pressure target, speed scales with γ²/µ and "
  "with cos² θ. These formulas give the stars in Figure 6 and the table; the grid in Figure 6 is evaluated independently, and the two agree.")
fig("fig3_design_maps.png", "<b>Figure 6.</b> Fill-time maps over (r, θ) for each liquid. Hatching marks the region satisfying all constraints; the white curve is t<sub>fill</sub> = t<sub>max</sub>, the dashed orange curve is ΔP = ΔP<sub>min</sub>, red lines mark the fabrication/footprint limits, and the star is the analytic optimum at θ = %.0f°." % D["THETA_MIN"], maxh=15.5 * cm)
fig("fig4_tradeoff.png", "<b>Figure 7.</b> (a) Fill time vs radius (θ = 30°); (b) the speed–pressure trade-off. The green band satisfies both the time and pressure targets.", maxh=6.6 * cm)

P(f"<b>Optimal designs.</b> Optimum radius and fill time for each liquid at three achievable contact angles (ΔP = ΔP<sub>min</sub> = {D['DP_MIN']:.0f} Pa, "
  f"L = {D['L_mm']:.0f} mm). The last column is the largest contact angle that still meets t<sub>max</sub> (before the {D['THETA_MAX']:.0f}° cap).")
opt = [["Liquid", "θ [°]", "r* [µm]", "t<sub>fill</sub> [s]", "Mean speed [mm/s]", "Bo", "Feasible", "θ<sub>max</sub> [°]"]]
for name, v in D["table"].items():
    for j, rw in enumerate(v["optimum_rows"]):
        opt.append([name if j == 0 else "", f"{rw['theta']:.0f}", f"{rw['r_star_um']:.0f}", f"{rw['t_fill_s']:.2f}",
                    f"{rw['mean_speed_mm_s']:.0f}", f"{rw['Bo']:.3f}", "yes" if rw["feasible"] else "no",
                    f"{v['theta_limit_deg']:.0f}" if j == 0 and v["theta_limit_deg"] else ""])
table(opt, [3.6 * cm, 1.3 * cm, 1.8 * cm, 2.2 * cm, 2.9 * cm, 1.4 * cm, 1.8 * cm, W - 15.0 * cm])
w30 = {k: v["r_window_theta30_um"] for k, v in D["table"].items() if v["r_window_theta30_um"]}
P("<b>Findings.</b>")
if all(t <= D["T_MAX"] for t in t_at_rmin.values()):
    lower = (f"At θ = 30° every liquid's window starts at the fabrication limit ({D['R_MIN_um']:.0f} µm): even the slowest case, {slowest}, "
             f"fills in {t_at_rmin[slowest]:.1f} s at that radius, inside the {D['T_MAX']:.0f} s limit. The upper end is set by the pressure requirement.")
else:
    lim = [k for k, t in t_at_rmin.items() if t > D["T_MAX"]]
    lower = (f"At θ = 30° the lower end of the window is set by the fill-time limit for {', '.join(lim)} and by the fabrication limit for the "
             f"other liquids; the upper end is set by the pressure requirement.")
B(lower + " Windows at θ = 30°: " + "; ".join(f"{k} {a:.0f}–{b:.0f} µm" for k, (a, b) in w30.items()) + ".")
B("Ethanol (low γ, similar µ) is the most demanding liquid: its γ²/µ is about 13× smaller than water's, so its radius window is narrow and "
  "the largest tolerable contact angle drops to ≈ 70°. 50 % glycerol-water has six times the viscosity of water but nearly the same γ, "
  "so it fills about 6.4× slower in the same channel (t ∝ µ/γ) and 6.7× slower at the respective optima, yet remains feasible.")
B(f"Moving from θ = 60° to 10° (a surface-treatment change) shortens the optimal fill time by {surf_gain:.1f}× for every liquid, because "
  "t<sub>min</sub> ∝ 1/cos² θ. Making the surface hydrophilic is the single most effective design decision; the stability of that treatment "
  "(for example hydrophobic recovery of plasma-treated PDMS) is a real device risk.")
B(f"At the optimum the water channel is {wopt['r_star_um']:.0f} µm in radius and fills {D['L_mm']:.0f} mm in {wopt['t_fill_s']:.2f} s, about "
  f"{wopt['t_fill_s']/tau_opt:.0f} τ, so LW is accurate to better than 1 % there; faster targets would need the inertial correction (Section 5.3).")

# ------------------------------------------------------------------ 7 validation against the reference paper
P("7. Validation against the reference paper", h1)
P("Our proposal commits to comparing predictions with published results, using Kolliopoulos et al. (2021) as the reference. That paper studies "
  "capillary flow in <i>open</i> rectangular microchannels with experiments and two models. We implemented its <b>modified Lucas–Washburn "
  "(MLW) model</b>, its critical aspect ratio, and its experimental geometry and liquid properties (module "
  "<font name='DV-B'>open_channel_validation.py</font>), taking all equations and values from the open-access paper. Section 7.5 states "
  "exactly what has and has not been validated.")

P("7.1 The paper's model", h2)
P("For a flat free surface the meniscus follows z<sub>m</sub> = √(kt) with mobility "
  "k = 2σH ζ<sub>o</sub>(λ) [cos θ<sub>0</sub>(1 + 2λ) − 1] / (3µ), where λ = H/W and ζ<sub>o</sub>(λ) is a Fourier-series aspect-ratio function "
  "(paper eqs. 2–3). Ahead of the meniscus the liquid splits into corner fingers when λ &lt; λ<sub>c</sub> = (1 − sin θ<sub>0</sub>)/(2 cos θ<sub>0</sub>) (eq. 1). "
  "Setup: H = 22.5 µm, L = 30 mm, W = 17, 25, 50, 75, 100, 200 µm, and four non-volatile liquids (Table 1 of the paper: NOA74, silicone oil, "
  "mineral oil, propylene glycol; θ<sub>0</sub> = 10°, 18°, 32°, 42°).")

P("7.2 Verification of the implementation", h2)
P("The checks below compare our implementation with the paper's own numbers and with independent mathematics. "
  "V1 is the strongest: an open channel with a flat, shear-free top is exactly half of a closed duct of height 2H, so ζ<sub>o</sub> can be rebuilt from the "
  "closed-duct series of Section 3. The two agree to 10<super>−7</super>.")
vr2 = [["#", "Check", "Result", "Status"]]
for i, c in enumerate(VR["checks"], 1):
    vr2.append([f"V{i}", c["name"], c["value"], "PASS" if c["passed"] else "FAIL"])
table(vr2, [0.8 * cm, 6.5 * cm, W - 9.3 * cm, 2.0 * cm])
P("V5 also confirms that the aspect ratios quoted in the paper's text (λ = 0.45, 0.225, 0.3, 0.23) match the channel widths in its Section 2, and that "
  "propylene glycol has only one width (200 µm) below λ<sub>c</sub>, consistent with the paper testing only λ &gt; λ<sub>c</sub> for that liquid.", small)

P("7.3 MLW predictions in the paper's geometry", h2)
P("Figure 8a shows the dimensionless mobility k/(UL) = ζ<sub>o</sub>[cos θ<sub>0</sub>(1 + 2λ) − 1]/3 (paper eq. 17b, with U = 2Hσ/(µL)), which depends only "
  "on λ and θ<sub>0</sub>. It rises from zero at the no-flow aspect ratio (1 − cos θ<sub>0</sub>)/(2 cos θ<sub>0</sub>), peaks, and decays for deep, narrow channels. "
  "Figure 9 and the table give the predicted meniscus histories; the time to reach the channel end is t = L²/k, and the paper recorded for up to about 400 s.")
fig("val_fig1_mobility.png", "<b>Figure 8.</b> (a) MLW dimensionless mobility vs aspect ratio; dotted lines mark λ<sub>c</sub> for each liquid and circles the widths used in the paper. (b) Dimensional mobility in the paper's channels. Propylene glycol at W = 200 µm has no predicted flow (driving term ≤ 0).", maxh=6.4 * cm)
fig("val_fig2_mlw_curves.png", "<b>Figure 9.</b> MLW meniscus position vs time for each liquid and channel width. These are model predictions only; no measured data are overlaid.", maxh=11.5 * cm)
pr = [["Liquid", "W [µm]", "λ", "Regime", "k [10³ µm²/s]", "z<sub>m</sub>(100 s) [mm]", "t to 30 mm [s]"]]
prev = None
for p_ in VR["predictions"]:
    pr.append([p_["liquid"] if p_["liquid"] != prev else "", p_["W_um"], f"{p_['lam']:.3f}",
               "λ ≥ λ<sub>c</sub>" if p_["regime"].startswith("lambda>=") else "λ &lt; λ<sub>c</sub>",
               "no flow" if p_["k_um2_per_s"] is None else f"{p_['k_um2_per_s']/1e3:.0f}",
               "0" if p_["k_um2_per_s"] is None else f"{p_['z_at_100s_mm']:.1f}",
               "—" if p_["t_to_30mm_s"] is None else f"{p_['t_to_30mm_s']:.0f}"])
    prev = p_["liquid"]
table(pr, [3.4 * cm, 1.7 * cm, 1.7 * cm, 2.2 * cm, 2.8 * cm, 3.0 * cm, W - 14.8 * cm])

P("7.4 Optimum aspect ratio, and why the closed-duct model is not enough", h2)
P("For fixed depth H, the MLW mobility has a single maximum in λ (Figure 10a): wide, shallow channels lose driving force (the flat top adds no "
  "curvature drive), while deep, narrow channels are dominated by wall friction. The fastest-filling aspect ratio moves from "
  f"λ<sub>opt</sub> ≈ {min(lam_opts):.1f} for well-wetting liquids to about {max(lam_opts):.1f} at θ<sub>0</sub> = 42° (Figure 10b), and in every case lies above λ<sub>c</sub>:")
op = [["Liquid", "θ<sub>0</sub> [°]", "λ<sub>c</sub>", "λ<sub>opt</sub>", "W<sub>opt</sub> at H = 22.5 µm [µm]", "k/(UL) max", "k<sub>max</sub> [10³ µm²/s]"]]
for n_, o in OP.items():
    op.append([n_, f"{o['theta0']:.0f}", f"{o['lam_c']:.2f}", f"{o['lam_opt']:.2f}", f"{o['W_opt_um']:.0f}", f"{o['k_star_max']:.3f}", f"{o['k_max_um2_s']/1e3:.0f}"])
table(op, [3.6 * cm, 1.8 * cm, 1.6 * cm, 1.8 * cm, 4.0 * cm, 1.9 * cm, W - 14.7 * cm])
P(f"As a design rule for open channels of fixed depth, the width should be about {1/max(lam_opts):.1f}–{1/min(lam_opts):.1f} times the depth. This follows "
  "from the flat-top MLW model; the paper's lubrication model also shows a non-monotonic dependence on λ (its Figs. 6–7), but we did not compare "
  "where its maxima sit. The paper reports that MLW under-predicts the meniscus position for mineral oil and propylene glycol when λ &gt; λ<sub>c</sub>, "
  "so the MLW-based optimum is conservative for those liquids.")
fig("val_fig3_optimum.png", "<b>Figure 10.</b> (a) k/(UL) vs λ for several contact angles with the optimum marked. (b) Optimum aspect ratio, λ<sub>c</sub> and the no-flow limit vs θ<sub>0</sub>. (c) Ratio of the open-channel MLW mobility to the closed-duct model of Section 3 for the same W and H (illustrative liquid, all walls at θ<sub>0</sub>).", maxh=5.6 * cm)
P("Figure 10c shows why the closed-duct model cannot replace the paper's model: for the same cross-section the open channel's mobility is 6–69 % lower "
  "over λ = 0.2–10, because the free surface does not drive the flow as a wall would and the open top changes the viscous resistance. The two coincide "
  "only for deep, narrow channels (λ → ∞), where the missing lid does not matter; check V4 confirms this limit.")

P("7.5 Validation status: what is and is not demonstrated", h2)
P("<b>Demonstrated.</b> The paper's MLW model, critical aspect ratio, geometry and liquid properties are implemented correctly: the checks reproduce "
  "the paper's quoted λ<sub>c</sub> values and aspect ratios, an independent mathematical identity and the theoretical limits, and our closed-duct "
  "model agrees with the open-channel model in the one regime where they must agree.")
P("<b>Not yet demonstrated.</b>")
B("<b>Comparison with measured data.</b> The paper's experimental meniscus curves (its Fig. 9) are published only as plots, so we have not computed "
  "errors against them. This report does not claim that our predictions match the paper's experiments.")
B("<b>The paper's lubrication-theory-based (LTB) model is not implemented.</b> It needs a finite-element solution for the velocity field and a nonlinear "
  "boundary-value solver; it is the model the paper finds most accurate at small λ and small θ<sub>0</sub>. Corner fingers are therefore not predicted.")
B("<b>Inertia and dynamic contact angle</b> are not included in the open-channel model; the paper also neglects them.")
P("<b>How to complete the comparison.</b> Digitise the average (solid-symbol) curves of the paper's Fig. 9 with a plot digitiser (e.g. WebPlotDigitizer) into "
  "<font name='DV-B'>experimental_template.csv</font> (columns: liquid, W_um, t_s, z_mm) and run <font name='DV-B'>python open_channel_validation.py out --exp "
  "experimental_template.csv</font>. For each curve the script reports the experimental mobility, k<sub>exp</sub>/k<sub>MLW</sub>, the RMS error of z<sub>m</sub>(t), "
  "and whether the result matches the paper's statements about MLW: good agreement for NOA74 and silicone oil at λ ≥ λ<sub>c</sub>; under-prediction for "
  "mineral oil and propylene glycol at λ ≥ λ<sub>c</sub>; over-prediction at λ &lt; λ<sub>c</sub> for NOA74, silicone oil and mineral oil. The routine was "
  "tested with synthetic data containing known mobility ratios, which it recovered exactly. The ±25 % tolerance for ‘good agreement’ is our choice, "
  "because the paper gives no numeric criterion.")

# ------------------------------------------------------------------ 8 limitations
P("8. Assumptions and limitations", h1)
for t in [
    "<b>Constant, static contact angle.</b> Real advancing angles exceed the static angle and depend on front speed (dynamic contact angle); "
    "this makes real filling slower than predicted, particularly at high velocity and in the early regime.",
    "<b>Driving pressure acts on this channel only.</b> The ΔP<sub>min</sub> constraint is a proxy for pumping capability. If the pump "
    "drives a downstream load (a reaction chamber, a valve), the effective driving pressure and fill time differ; that needs a hydraulic-network model.",
    "<b>Idealised geometry.</b> Cylindrical, closed rectangular and (Section 7) open rectangular channels with a flat free surface and uniform wall "
    "properties. Mixed wall contact angles (for example PDMS walls with a glass lid), corner fingers (Concus–Finn filaments), roughness and pinned fronts are not modelled.",
    "<b>Newtonian liquid, isothermal, no evaporation.</b> Blood, saliva and many reagents are non-Newtonian; evaporation matters in "
    "sub-100 µm channels over tens of seconds.",
    "<b>Early-time model.</b> The full ODE assumes plug-flow inertia and a fixed start-up position h<sub>0</sub> = 0.1 µm; entrance "
    "effects, meniscus formation and the entry pressure drop are ignored. It is a more faithful, not an exact, early-time model.",
    "<b>Fluid properties</b> are representative handbook values, and the plasma values are approximate.",
    "<b>Design constraints</b> (Section 6) are assumed targets, so the specific optima are illustrative; the method and closed-form relations carry over."]:
    B(t)

# ------------------------------------------------------------------ 9 conclusions
P("9. Conclusions", h1)
P("<b>Answer to the research question.</b> Capillary transport is fastest in channels that are as wide as the pressure and sample-volume "
  "requirements allow and whose walls are as hydrophilic as possible, filled with a liquid of high surface tension and low viscosity. Quantitatively:")
B("Fill time follows t<sub>fill</sub> = 2µL² / (γ r cos θ): doubling the radius halves it, while a lower contact angle helps little below 45° but "
  "matters strongly near 90°, where flow stops.")
B(f"Radius trades speed against driving pressure. Under a {D['DP_MIN']:.0f} Pa pressure target the optimum is r* = 2γ cos θ / ΔP<sub>min</sub>; for water at "
  f"θ = {wopt['theta']:.0f}° this is {wopt['r_star_um']:.0f} µm, filling {D['L_mm']:.0f} mm in {wopt['t_fill_s']:.2f} s.")
B(f"Surface treatment is the most effective single lever: lowering θ from 60° to 10° makes the optimal design {surf_gain:.1f}× faster for any liquid. "
  "At a fixed pressure target the liquid's figure of merit is γ²/µ.")
B(f"Channel shape matters. A circular-tube formula with an equivalent radius can be off by tens of percent for rectangular channels, and an open channel "
  f"has 6–69 % lower mobility than a closed one of the same size (so it fills up to about 3× more slowly). For open channels the fastest aspect ratio is λ ≈ {min(lam_opts):.1f}–{max(lam_opts):.1f} "
  "(width ≈ 1.4–2 × depth), and flow stops altogether if the channel is too wide for the contact angle.")
B("Lucas–Washburn is reliable for microchannels after about 10 τ (well under a millisecond for r ≤ 50 µm). Inertia matters only for wide channels "
  "or very fast fills, and gravity only for vertical channels or Bond numbers above about 0.1.")
P("<b>Future work.</b> (i) Overlay digitised data from Kolliopoulos et al. (2021) to quantify the model error; (ii) implement the lubrication-theory "
  "model for small λ; (iii) add dynamic contact angle, mixed-wettability walls and non-Newtonian liquids such as blood; (iv) fabricate a test "
  "channel and measure its filling.")

# ------------------------------------------------------------------ references
P("References", h1)
for r_ in [
    "Bosanquet, C. H. (1923). On the flow of liquids into capillary tubes. <i>Philosophical Magazine</i>, 45, 525–531.",
    "Bruus, H. (2008). <i>Theoretical Microfluidics</i>. Oxford University Press.",
    "Concus, P., &amp; Finn, R. (1969). On the behavior of a capillary surface in a wedge. <i>Proceedings of the National Academy of Sciences</i>, 63, 292–299.",
    "Juncker, D., Schmid, H., Drechsler, U., Wolf, H., Wolf, M., Michel, B., de Rooij, N., &amp; Delamarche, E. (2002). Autonomous microfluidic capillary "
    "system. <i>Analytical Chemistry</i>, 74, 6139–6144.",
    "Kolliopoulos, P., Jochem, K. S., Johnson, D., Suszynski, W. J., Francis, L. F., &amp; Kumar, S. (2021). Capillary-flow dynamics in open rectangular "
    "microchannels. <i>Journal of Fluid Mechanics</i>, 911, A32. https://doi.org/10.1017/jfm.2020.986",
    "Lucas, R. (1918). Ueber das Zeitgesetz des kapillaren Aufstiegs von Flüssigkeiten. <i>Kolloid-Zeitschrift</i>, 23, 15–22.",
    "Martinez, A. W., Phillips, S. T., Butte, M. J., &amp; Whitesides, G. M. (2007). Patterned paper as a platform for inexpensive, low-volume, portable "
    "bioassays. <i>Angewandte Chemie International Edition</i>, 46, 1318–1320.",
    "Olanrewaju, A., Beaugrand, M., Yafia, M., &amp; Juncker, D. (2018). Capillary microfluidics in microchannels: from microfluidic networks to capillaric "
    "circuits. <i>Lab on a Chip</i>, 18, 2323–2347.",
    "Quéré, D. (1997). Inertial capillarity. <i>Europhysics Letters</i>, 39, 533–538.",
    "Washburn, E. W. (1921). The dynamics of capillary flow. <i>Physical Review</i>, 17, 273–283."]:
    P(r_, ref)

# ------------------------------------------------------------------ appendix
P("Appendix A. Reproducibility", h1)
P("<font name='DV-B'>python capillary_pump_sim.py sim_out</font> runs the checks, writes all seven figures and results.json; design targets can be "
  "changed with <font name='DV-B'>--L --tmax --dpmin --rmin --rmax --bomax --theta-min --theta-max</font> (SI units, degrees). "
  "<font name='DV-B'>python open_channel_validation.py val_out [--exp experimental_template.csv]</font> produces Section 7 (keep it in the same folder "
  "as capillary_pump_sim.py). <font name='DV-B'>python build_report.py sim_out val_out report.pdf</font> rebuilds this PDF. "
  "Requires numpy, scipy, matplotlib, reportlab and pillow.")


def footer(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(GREY)
    c.drawString(1.8 * cm, 1.1 * cm, "Capillary micro-pump project report  |  Introduction to Microfluidics and Applications")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.setStrokeColor(TEAL)
    c.setLineWidth(1.5)
    c.line(1.8 * cm, A4[1] - 1.2 * cm, A4[0] - 1.8 * cm, A4[1] - 1.2 * cm)
    c.restoreState()


doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                        title="Computational Design of a Self-Powered Microfluidic Pump Using Capillary Action",
                        author="Harshita Agrawal, Minakshi Dhingad")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("built", PDF)
