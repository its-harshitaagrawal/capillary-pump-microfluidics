Computational Design of a Self-Powered Microfluidic Pump Using Capillary Action
Harshita Agrawal (2023CH10969), Minakshi Dhingad (2023CH70817)

Capillary_Pump_Project_Report.pdf   final report (12 pages)
code/capillary_pump_sim.py          main simulation: Lucas-Washburn, inertia + gravity ODE,
                                    rectangular ducts, parametric design study (8 checks)
code/open_channel_validation.py     reference-paper model (Kolliopoulos et al. 2021), 6 checks
code/build_report.py                rebuilds the PDF from the simulation outputs
code/experimental_template.csv      fill with digitised points from Fig. 9 of the paper (optional)
code/requirements.txt               Python packages needed

How to run (inside the code folder):
    pip install -r requirements.txt
    python capillary_pump_sim.py sim_out
    python open_channel_validation.py val_out
    python build_report.py sim_out val_out report.pdf
