"""
Assignment 5b
=============
Finite capacity, realized demand.
Evaluate the fixed plan from Assignment 5a.

The fixed 5a plan contains:
  - production quantities p[i,t]
  - setup decisions y[i,t]
  - weekly overtime ox[t], oy[t]
  - permanent modernization dx, dy

Backorders are allowed and charged at EUR 250 per unit per period.

Output: output_5b.xlsx
"""

import importlib.util
from pathlib import Path

import pandas as pd
from gurobipy import GRB
from utils import (
    build_base_model,
    add_overtime_vars, add_modernization_vars,
    add_capacity_combined, set_combined_objective,
    compute_service_metrics,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)


# -----------------------------------------------------------------------------
# Load 5aFUNCTION.py
# -----------------------------------------------------------------------------
def load_assignment_5a_module():
    module_path = Path(__file__).with_name("5aFUNCTION.py")
    spec = importlib.util.spec_from_file_location("assignment5a_module", module_path)

    if spec is None or spec.loader is None:
        raise ImportError("Could not load module from 5aFUNCTION.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Step 1: solve 5a and retrieve the fixed plan
# -----------------------------------------------------------------------------
assignment5a = load_assignment_5a_module()

plan_5a = assignment5a.solve_5a_plan(
    write_output=False,
    output_filename="output_5a.xlsx",
    print_summary=False
)

data    = plan_5a["data"]
parts   = plan_5a["parts"]
periods = plan_5a["periods"]
D_fcst  = plan_5a["D_fcst"]

p_fix   = plan_5a["p_fix"]
y_fix   = plan_5a["y_fix"]
ox_fix  = plan_5a["ox_fix"]
oy_fix  = plan_5a["oy_fix"]
dx_fix  = plan_5a["dx_fix"]
dy_fix  = plan_5a["dy_fix"]

D_real  = data["D_real"]

BO_COST        = data["BO_COST"]
OT_COST_X      = data["OT_COST_X"]
OT_COST_Y      = data["OT_COST_Y"]
MOD_COST_X     = data["MOD_COST_X"]
MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
CAP_X          = data["CAP_X"]
CAP_Y          = data["CAP_Y"]


# -----------------------------------------------------------------------------
# Step 2: evaluate the fixed 5a plan under realized demand
# -----------------------------------------------------------------------------
m, p, q, y, b = build_base_model(
    data, D_real, "Assignment_5b", with_backorders=True
)

ox, oy = add_overtime_vars(m, data)
dx, dy = add_modernization_vars(m, data)

set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data, with_backorders=True)
add_capacity_combined(m, p, ox, oy, dx, dy, data)

# Make constraint names available for lookup
m.update()

# Remove final-backorder-clearing constraint.
# In 5b we evaluate a fixed 5a plan, so backlog may remain at the end.
c_final_bo = m.getConstrByName("no_final_backorder")
if c_final_bo is not None:
    m.remove(c_final_bo)
    m.update()

# Fix all decisions from 5a
for i in parts:
    for t in periods:
        m.addConstr(p[i, t] == p_fix[i, t], name="fix_p_" + i + "_" + str(t))
        m.addConstr(y[i, t] == y_fix[i, t], name="fix_y_" + i + "_" + str(t))

for t in periods:
    m.addConstr(ox[t] == ox_fix[t], name="fix_ox_" + str(t))
    m.addConstr(oy[t] == oy_fix[t], name="fix_oy_" + str(t))

m.addConstr(dx == dx_fix, name="fix_dx")
m.addConstr(dy == dy_fix, name="fix_dy")

m.optimize()


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

    # Cost components
    total_ot_x   = sum(OT_COST_X * ox[t].X for t in periods)
    total_ot_y   = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods)
    mod_cost_x   = MOD_COST_X * dx.X
    mod_cost_y   = MOD_COST_PCT_Y * dy.X
    total_bo     = sum(BO_COST * b[t].X for t in periods)

    new_cap_x    = CAP_X + dx.X
    new_cap_y    = CAP_Y * (1.0 + dy.X / 100.0)

    extra = {
        "Overtime Cost X (EUR)":      total_ot_x,
        "Overtime Cost Y (EUR)":      total_ot_y,
        "Modernization Cost X (EUR)": mod_cost_x,
        "Modernization Cost Y (EUR)": mod_cost_y,
        "Backorder Cost (EUR)":       total_bo,
    }

    df_cost, total_setup, total_holding = build_cost_summary(
        p, q, y, data, extra=extra
    )

    total_cost = (
        total_setup + total_holding + total_ot_x + total_ot_y
        + mod_cost_x + mod_cost_y + total_bo
    )
    df_cost.loc["TOTAL", "Grand Total (EUR)"] = round(total_cost, 2)

    service_level, fill_rate = compute_service_metrics(b, D_real, periods)

    print_cost_summary("ASSIGNMENT 5B — Fixed 5a Plan | Realized Demand", df_cost)
    print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
    print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
    print("  Modernization WS-X : +" + "{:.1f}".format(dx.X) + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
    print("  Modernization WS-Y : +" + "{:.4f}".format(dy.X) + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
    print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
    print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
    print("  Backorder Cost     : EUR " + "{:,.2f}".format(total_bo))
    print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))
    print("  Service Level      :  " + "{:.2%}".format(service_level))
    print("  Fill Rate          :  " + "{:.2%}".format(fill_rate))

    # Overtime schedule
    ot_rows = []
    for t in periods:
        ot_rows.append({
            "Period":          t,
            "OT Units X":      round(ox[t].X, 1),
            "OT Minutes Y":    round(oy[t].X, 1),
            "OT Hours Y":      round(oy[t].X / 60.0, 2),
            "OT Cost X (EUR)": round(OT_COST_X * ox[t].X, 2),
            "OT Cost Y (EUR)": round(OT_COST_Y * (oy[t].X / 60.0), 2),
        })
    df_ot = pd.DataFrame(ot_rows).set_index("Period")

    # Summary metrics
    summary_rows = [
        {"Metric": "Setup Cost (EUR)",            "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",          "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR)",       "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR)",       "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)",  "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",  "Value": round(mod_cost_y, 2)},
        {"Metric": "Backorder Cost (EUR)",        "Value": round(total_bo, 2)},
        {"Metric": "Total Cost (EUR)",            "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)", "Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",     "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",   "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",     "Value": round(new_cap_y, 2)},
        {"Metric": "Service Level",               "Value": round(service_level, 4)},
        {"Metric": "Fill Rate",                   "Value": round(fill_rate, 4)},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

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

    write_excel("output_5b.xlsx", {
        "Summary":         df_summary,
        "Cost per Part":   df_cost,
        "Overtime":        df_ot,
        "Production Plan": pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
        "Backorders":      df_bo,
    })

else:
    print("Model not solved. Gurobi status: " + str(m.Status))