"""
Assignment 2a
=============
Finite capacity, forecasted demand.
WS-X : E2801 <= 800 units/week
WS-Y : 3*B1401 + 2*B2302 <= 10 000 min/week  (7*24*60 - 80 maintenance)
No backorders.

Output: output_2a.xlsx
"""

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model, add_capacity_constraints,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
data    = load_data()
parts   = data["parts"]
periods = data["periods"]
D       = data["D_fcst"]
CAP_X   = data["CAP_X"]
CAP_Y   = data["CAP_Y"]
PROC_Y  = data["PROC_Y"]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
m, p, q, y, _ = build_base_model(data, D, "Assignment_2a", with_backorders=False)

add_capacity_constraints(m, p, data)

m.optimize()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

    df_cost, total_setup, total_holding = build_cost_summary(p, q, y, data)
    print_cost_summary("ASSIGNMENT 2a — Finite Capacity | Forecasted Demand", df_cost)

    # Workstation utilisation per period
    util_rows = []
    for t in periods:
        x_used = p["E2801", t].X
        y_used = PROC_Y["B1401"] * p["B1401", t].X + PROC_Y["B2302"] * p["B2302", t].X
        util_rows.append({
            "Period":           t,
            "WS-X Used (units)": round(x_used, 1),
            "WS-X Cap (units)":  CAP_X,
            "WS-X Util (%)":     round(x_used / CAP_X * 100, 1),
            "WS-Y Used (min)":   round(y_used, 1),
            "WS-Y Cap (min)":    CAP_Y,
            "WS-Y Util (%)":     round(y_used / CAP_Y * 100, 1),
        })
    df_util = pd.DataFrame(util_rows).set_index("Period")

    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)
    df_dem   = demand_row_df(D, periods)

    write_excel("output_2a.xlsx", {
        "Cost Summary":    df_cost,
        "WS Utilisation":  df_util,
        "Production Plan": pd.concat([df_prod, df_dem]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
    })

else:
    print("Model not solved. Gurobi status: " + str(m.Status))