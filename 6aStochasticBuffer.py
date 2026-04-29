"""
Assignment 6
============
Start from the model in assignment 5 and improve performance under real demand
by adding a stochastic buffer for the end product E2801.

Approach
--------
1. Baseline plan: reuse the fixed plan from Assignment 5a
2. Improved plan: solve the Assignment 5 model again, but with a safety stock
   constraint on E2801
3. Control step: evaluate both fixed plans under realized demand

Chosen stochastic buffer
------------------------
95% service-level safety stock for E2801:
    buffer = 42 units

Output
------
- output_6a.xlsx                : buffered plan under forecast demand
- output_6_baseline_control.xlsx: fixed 5a plan evaluated on realized demand
- output_6_buffer_control.xlsx  : fixed 6a plan evaluated on realized demand
- output_6_comparison.xlsx      : side-by-side comparison
"""

import importlib.util
from pathlib import Path

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data,
    build_base_model,
    add_overtime_vars,
    add_modernization_vars,
    add_capacity_combined,
    set_combined_objective,
    compute_service_metrics,
    make_plan_df,
    make_setup_df,
    build_cost_summary,
    demand_row_df,
    write_excel,
    print_cost_summary,
)

# -------------------------------------------------------------------------
# Parameters for Assignment 6
# -------------------------------------------------------------------------
BUFFER_UNITS = 59   # 95% stochastic safety stock for E2801
# If you want bias-corrected uplift instead, 45 is also defensible.


# -------------------------------------------------------------------------
# Load 5aFUNCTION.py
# -------------------------------------------------------------------------
def load_assignment_5a_module():
    module_path = Path(__file__).with_name("5aFUNCTION.py")
    spec = importlib.util.spec_from_file_location("assignment5a_module", module_path)

    if spec is None or spec.loader is None:
        raise ImportError("Could not load module from 5aFUNCTION.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------------------
# Solve buffered 6a plan
# -------------------------------------------------------------------------
def solve_6a_buffered_plan(buffer_units=42, write_output=True, output_filename="output_6a.xlsx", print_summary=True):
    """
    Solve the Assignment 5 model again, but protect the end product E2801
    with a stochastic safety stock.

    Returns a fixed plan dictionary, analogous to 5aFUNCTION.py.
    """

    data           = load_data()
    parts          = data["parts"]
    periods        = data["periods"]
    demand         = data["D_fcst"]

    BO_COST        = data["BO_COST"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]

    # --- Build model ---
    m, p, q, y, _ = build_base_model(
        data, demand, "Assignment_6a_buffered", with_backorders=False
    )

    ox, oy = add_overtime_vars(m, data)
    dx, dy = add_modernization_vars(m, data)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, None, data, with_backorders=False)
    add_capacity_combined(m, p, ox, oy, dx, dy, data)

    # Stochastic safety stock for end product E2801
    for t in periods:
        m.addConstr(q["E2801", t] >= buffer_units, name="ss_E2801_" + str(t))

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError("6a model not solved. Gurobi status: " + str(m.Status))

    # --- Cost components ---
    total_ot_x   = sum(OT_COST_X * ox[t].X for t in periods)
    total_ot_y   = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods)
    mod_cost_x   = MOD_COST_X * dx.X
    mod_cost_y   = MOD_COST_PCT_Y * dy.X
    new_cap_x    = CAP_X + dx.X
    new_cap_y    = CAP_Y * (1.0 + dy.X / 100.0)

    extra = {
        "Overtime Cost X (EUR)":      total_ot_x,
        "Overtime Cost Y (EUR)":      total_ot_y,
        "Modernization Cost X (EUR)": mod_cost_x,
        "Modernization Cost Y (EUR)": mod_cost_y,
    }

    df_cost, total_setup, total_holding = build_cost_summary(
        p, q, y, data, extra=extra
    )

    total_cost = total_setup + total_holding + total_ot_x + total_ot_y + mod_cost_x + mod_cost_y
    df_cost.loc["TOTAL", "Grand Total (EUR)"] = round(total_cost, 2)

    if print_summary:
        print_cost_summary("ASSIGNMENT 6A — Buffered Forecast Plan", df_cost)
        print("  Buffer E2801       :  " + str(buffer_units) + " units")
        print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
        print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
        print("  Modernization WS-X : +" + "{:.1f}".format(dx.X) + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
        print("  Modernization WS-Y : +" + "{:.4f}".format(dy.X) + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
        print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
        print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
        print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))

    # --- Overtime schedule ---
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

    summary_rows = [
        {"Metric": "Buffer E2801 (units)",       "Value": round(buffer_units, 2)},
        {"Metric": "Setup Cost (EUR)",           "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",         "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR)",      "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR)",      "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)", "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)", "Value": round(mod_cost_y, 2)},
        {"Metric": "Total Cost (EUR)",           "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)","Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",    "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",  "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",    "Value": round(new_cap_y, 2)},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)
    df_dem   = demand_row_df(demand, periods, label="Forecast Demand E2801")

    if write_output:
        write_excel(output_filename, {
            "Summary":         df_summary,
            "Cost per Part":   df_cost,
            "Overtime":        df_ot,
            "Production Plan": pd.concat([df_prod, df_dem]),
            "Inventory Plan":  df_inv,
            "Setup Decisions": df_setup,
        })

    # Fixed plan for realized-demand evaluation
    p_fix  = {(i, t): int(round(p[i, t].X)) for i in parts for t in periods}
    y_fix  = {(i, t): int(round(y[i, t].X)) for i in parts for t in periods}
    ox_fix = {t: float(ox[t].X) for t in periods}
    oy_fix = {t: float(oy[t].X) for t in periods}
    dx_fix = float(dx.X)
    dy_fix = float(dy.X)

    return {
        "data": data,
        "parts": parts,
        "periods": periods,
        "D_fcst": demand,
        "buffer_units": buffer_units,
        "p_fix": p_fix,
        "y_fix": y_fix,
        "ox_fix": ox_fix,
        "oy_fix": oy_fix,
        "dx_fix": dx_fix,
        "dy_fix": dy_fix,
    }


# -------------------------------------------------------------------------
# Evaluate a fixed plan under realized demand
# -------------------------------------------------------------------------
def evaluate_fixed_plan_under_real_demand(plan, label, output_filename):
    """
    Evaluate a fixed plan under realized demand, with backorders allowed.
    This is the control step.
    """

    data    = plan["data"]
    parts   = plan["parts"]
    periods = plan["periods"]
    D_fcst  = plan["D_fcst"]
    D_real  = data["D_real"]

    p_fix   = plan["p_fix"]
    y_fix   = plan["y_fix"]
    ox_fix  = plan["ox_fix"]
    oy_fix  = plan["oy_fix"]
    dx_fix  = plan["dx_fix"]
    dy_fix  = plan["dy_fix"]

    BO_COST        = data["BO_COST"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]
    PROC_Y         = data["PROC_Y"]

    # --- Build evaluation model ---
    m, p, q, y, b = build_base_model(
        data, D_real, "Eval_" + label, with_backorders=True
    )

    ox, oy = add_overtime_vars(m, data)
    dx, dy = add_modernization_vars(m, data)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data, with_backorders=True)
    add_capacity_combined(m, p, ox, oy, dx, dy, data)

    m.update()

    # Fixed-plan evaluation: backlog may remain at end of horizon
    c_final_bo = m.getConstrByName("no_final_backorder")
    if c_final_bo is not None:
        m.remove(c_final_bo)
        m.update()

    # Fix all decisions from source plan
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

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError("Evaluation model not solved for " + label + ". Status: " + str(m.Status))

    # --- Cost components ---
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

    print_cost_summary("CONTROL — " + label, df_cost)
    print("  Backorder Cost     : EUR " + "{:,.2f}".format(total_bo))
    print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))
    print("  Service Level      :  " + "{:.2%}".format(service_level))
    print("  Fill Rate          :  " + "{:.2%}".format(fill_rate))

    # --- Overtime schedule ---
    ot_rows = []
    for t in periods:
        x_used = p["E2801", t].X
        y_used = PROC_Y["B1401"] * p["B1401", t].X + PROC_Y["B2302"] * p["B2302", t].X
        ot_rows.append({
            "Period":            t,
            "OT Units X":        round(ox[t].X, 1),
            "OT Minutes Y":      round(oy[t].X, 1),
            "OT Hours Y":        round(oy[t].X / 60.0, 2),
            "WS-X Used (units)": round(x_used, 1),
            "WS-X Cap (units)":  round(new_cap_x + ox[t].X, 1),
            "WS-Y Used (min)":   round(y_used, 1),
            "WS-Y Cap (min)":    round(new_cap_y + oy[t].X, 1),
        })
    df_ot = pd.DataFrame(ot_rows).set_index("Period")

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
        {"Metric": "Service Level",               "Value": round(service_level, 4)},
        {"Metric": "Fill Rate",                   "Value": round(fill_rate, 4)},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)
    df_dem_fcst = demand_row_df(D_fcst, periods, label="Forecast Demand E2801")
    df_dem_real = demand_row_df(D_real, periods, label="Realized Demand E2801")

    bo_row = {"Part": "Backorder E2801"}
    for t in periods:
        bo_row["W" + str(t)] = int(round(b[t].X))
    df_bo = pd.DataFrame([bo_row]).set_index("Part")

    write_excel(output_filename, {
        "Summary":         df_summary,
        "Cost per Part":   df_cost,
        "Overtime+Cap":    df_ot,
        "Production Plan": pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
        "Backorders":      df_bo,
    })

    return {
        "Label": label,
        "Setup Cost": total_setup,
        "Holding Cost": total_holding,
        "OT Cost X": total_ot_x,
        "OT Cost Y": total_ot_y,
        "Mod Cost X": mod_cost_x,
        "Mod Cost Y": mod_cost_y,
        "Backorder Cost": total_bo,
        "Total Cost": total_cost,
        "Service Level": service_level,
        "Fill Rate": fill_rate,
        "dx": dx.X,
        "dy": dy.X,
    }


# -------------------------------------------------------------------------
# Run Assignment 6
# -------------------------------------------------------------------------
# Baseline from 5a
assignment5a = load_assignment_5a_module()
baseline_5a_plan = assignment5a.solve_5a_plan(
    write_output=False,
    output_filename="output_5a.xlsx",
    print_summary=False
)

# Improved buffered plan
buffered_6a_plan = solve_6a_buffered_plan(
    buffer_units=BUFFER_UNITS,
    write_output=True,
    output_filename="output_6a.xlsx",
    print_summary=True
)

# Control evaluations on realized demand
baseline_control = evaluate_fixed_plan_under_real_demand(
    baseline_5a_plan,
    label="5A baseline on realized demand",
    output_filename="output_6_baseline_control.xlsx"
)

buffer_control = evaluate_fixed_plan_under_real_demand(
    buffered_6a_plan,
    label="6A buffered plan on realized demand",
    output_filename="output_6_buffer_control.xlsx"
)

# Comparison table
df_compare = pd.DataFrame([
    {"Plan": "5A baseline", **baseline_control},
    {"Plan": "6A buffered", **buffer_control},
]).set_index("Plan")

write_excel("output_6_comparison.xlsx", {
    "Comparison": df_compare
})