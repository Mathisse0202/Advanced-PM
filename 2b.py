"""
Assignment 2b
=============
Finite capacity, realized demand.
Evaluate the fixed production plan from Assignment 2a.

WS-X : E2801 <= 800 units/week
WS-Y : 3*B1401 + 2*B2302 <= 10 000 min/week  (7*24*60 - 80 maintenance)
Backorders allowed.

Output: output_2b.xlsx
"""

import importlib.util
from pathlib import Path

import pandas as pd
from gurobipy import GRB
from utils import (
    build_base_model,
    add_capacity_constraints,
    compute_service_metrics,
    make_plan_df,
    make_setup_df,
    build_cost_summary,
    demand_row_df,
    write_excel,
    print_cost_summary,
)


# -----------------------------------------------------------------------------
# Load 2aFUNCTION.py
# -----------------------------------------------------------------------------
def load_assignment_2a_module():
    """
    Load 2aFUNCTION.py as a Python module.

    This is necessary because the filename starts with a digit,
    so a normal import statement cannot be used.
    """
    module_path = Path(__file__).with_name("2aFUNCTION.py")
    spec = importlib.util.spec_from_file_location("assignment2a_module", module_path)

    if spec is None or spec.loader is None:
        raise ImportError("Could not load module from 2aFUNCTION.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Step 1: solve Assignment 2a and retrieve the fixed plan
# -----------------------------------------------------------------------------
assignment2a = load_assignment_2a_module()

plan_2a = assignment2a.solve_2a_plan(
    write_output=False,
    output_filename="output_2a.xlsx",
    print_summary=False
)

data    = plan_2a["data"]
parts   = plan_2a["parts"]
periods = plan_2a["periods"]
D_fcst  = plan_2a["D_fcst"]
p_fix   = plan_2a["p_fix"]
y_fix   = plan_2a["y_fix"]

# Realized demand for 2b evaluation
D_real  = data["D_real"]

# Capacity and cost parameters
CAP_X   = data["CAP_X"]
CAP_Y   = data["CAP_Y"]
PROC_Y  = data["PROC_Y"]
BO_COST = data["BO_COST"]


# -----------------------------------------------------------------------------
# Step 2: evaluate the fixed 2a plan under realized demand
# -----------------------------------------------------------------------------
m, p, q, y, b = build_base_model(
    data, D_real, "Assignment_2b", with_backorders=True
)

# Same finite-capacity constraints as in 2a
add_capacity_constraints(m, p, data)

# Make constraint names available for lookup
m.update()

# Remove final-backorder-clearing constraint.
# In 2b we evaluate a fixed plan, so backlog may remain at the end.
c_final_bo = m.getConstrByName("no_final_backorder")
if c_final_bo is not None:
    m.remove(c_final_bo)
    m.update()

# Fix production quantities and setup decisions to the 2a solution
for i in parts:
    for t in periods:
        m.addConstr(p[i, t] == p_fix[i, t], name="fix_p_" + i + "_" + str(t))
        m.addConstr(y[i, t] == y_fix[i, t], name="fix_y_" + i + "_" + str(t))

m.optimize()


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

    service_level, fill_rate = compute_service_metrics(b, D_real, periods)
    backorder_cost = sum(BO_COST * b[t].X for t in periods)

    df_cost, total_setup, total_holding = build_cost_summary(
        p, q, y, data,
        extra={"Backorder Cost (EUR)": backorder_cost}
    )
    df_cost.loc["TOTAL", "Grand Total (EUR)"] = round(
        total_setup + total_holding + backorder_cost, 2
    )

    print_cost_summary("ASSIGNMENT 2b — Fixed 2a Plan | Realized Demand", df_cost)
    print(f"Service level: {service_level:.4f}")
    print(f"Fill rate:     {fill_rate:.4f}")

    # Workstation utilisation per period
    util_rows = []
    for t in periods:
        x_used = p["E2801", t].X
        y_used = (
            PROC_Y["B1401"] * p["B1401", t].X
            + PROC_Y["B2302"] * p["B2302", t].X
        )
        util_rows.append({
            "Period":            t,
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

    df_dem_fcst = demand_row_df(D_fcst, periods, label="Forecast Demand E2801")
    df_dem_real = demand_row_df(D_real, periods, label="Realized Demand E2801")

    # Backorders per period
    bo_row = {"Part": "Backorder E2801"}
    for t in periods:
        bo_row["W" + str(t)] = int(round(b[t].X))
    df_bo = pd.DataFrame([bo_row]).set_index("Part")

    # Service metrics
    df_service = pd.DataFrame([
        {"Metric": "Service Level",        "Value": round(service_level, 4)},
        {"Metric": "Fill Rate",            "Value": round(fill_rate, 4)},
        {"Metric": "Backorder Cost (EUR)", "Value": round(backorder_cost, 2)},
        {"Metric": "Total Real Demand",    "Value": int(sum(D_real))},
    ]).set_index("Metric")

    write_excel("output_2b.xlsx", {
        "Cost Summary":    df_cost,
        "Service Metrics": df_service,
        "WS Utilisation":  df_util,
        "Production Plan": pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
        "Backorders":      df_bo,
    })

else:
    print("Model not solved. Gurobi status: " + str(m.Status))