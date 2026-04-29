"""
assignment_3a.py
----------------
Assignment 3a: Production planning with finite capacity + OVERTIME (forecasted demand).

Workstation X: capacity 800 units/week, overtime up to 300 units/week at €2/unit
Workstation Y: capacity 10000 min/week, overtime up to 38 hours/week at €120/hour

Uses forecasted demand (D_fcst).
"""

import pandas as pd
from datetime import datetime

from utils import (
    load_data,
    build_base_model,
    add_overtime_vars,
    add_capacity_with_overtime,
    set_overtime_objective,
    make_plan_df,
    make_setup_df,
    build_cost_summary,
    demand_row_df,
    write_excel,
    print_cost_summary,
)


def _safe_write(filename, sheets):
    """Write Excel; if file is locked, fall back to a timestamped name."""
    try:
        write_excel(filename, sheets)
    except PermissionError:
        ts = datetime.now().strftime("%H%M%S")
        fallback = filename.replace(".xlsx", f"_{ts}.xlsx")
        print(f"  [Warning] {filename} is open in Excel — writing to {fallback} instead.")
        write_excel(fallback, sheets)


def solve_3a(data):
    demand  = data["D_fcst"]
    parts   = data["parts"]
    periods = data["periods"]

    # Build base model (no backorders for part a)
    m, p, q, y, b = build_base_model(
        data, demand,
        model_name="Assignment3a_Overtime_Forecast",
        with_backorders=False,
    )

    # Add overtime variables
    ox, oy = add_overtime_vars(m, data)

    # Add capacity constraints extended with overtime
    add_capacity_with_overtime(m, p, ox, oy, data)

    # Set full objective: setup + holding + overtime costs
    set_overtime_objective(m, p, q, y, ox, oy, b, data, with_backorders=False)

    m.optimize()

    if m.Status not in (2, 9):  # 2 = OPTIMAL, 9 = TIME_LIMIT with solution
        print("No feasible solution found. Gurobi status:", m.Status)
        return

    # -------------------------------------------------------------------------
    # Overtime costs (clamp tiny negatives from floating point to 0)
    OT_COST_X = data["OT_COST_X"]
    OT_COST_Y = data["OT_COST_Y"]
    ot_x_vals = {t: max(ox[t].X, 0.0) for t in periods}
    ot_y_vals = {t: max(oy[t].X, 0.0) for t in periods}

    ot_cost_x_total = sum(OT_COST_X * ot_x_vals[t] for t in periods)
    # Sum all minutes first, then convert once — matches model objective exactly
    total_oy_min    = sum(ot_y_vals[t] for t in periods)
    ot_cost_y_total = OT_COST_Y * (total_oy_min / 60.0)
    ot_total = ot_cost_x_total + ot_cost_y_total

    # -------------------------------------------------------------------------
    # Cost summary — extra cols only on TOTAL row; fill part rows with ""
    extra = {
        "OT Cost WS-X (EUR)":        ot_cost_x_total,
        "OT Cost WS-Y (EUR)":        ot_cost_y_total,
        "Overtime Cost Total (EUR)":  ot_total,
    }
    df_cost, total_setup, total_holding = build_cost_summary(p, q, y, data, extra=extra)

    # build_cost_summary uses raw q[i,t].X which Gurobi returns as floats like
    # 24.9999997 even though q is INTEGER. Recompute holding cost with proper
    # rounding so reported values match the true integer solution.
    HC = data["HC"]
    total_holding_rounded = sum(
        HC[i] * round(q[i, t].X) for i in parts for t in periods
    )
    holding_diff = total_holding_rounded - total_holding  # typically +/- a few cents
    # Patch each part row
    for i in parts:
        h_rounded = sum(HC[i] * round(q[i, t].X) for t in periods)
        s_val = df_cost.loc[i, "Setup Cost (EUR)"]
        df_cost.loc[i, "Holding Cost (EUR)"] = round(h_rounded, 2)
        df_cost.loc[i, "Total Cost (EUR)"]   = round(float(s_val) + h_rounded, 2)
    # Patch TOTAL row
    df_cost.loc["TOTAL", "Holding Cost (EUR)"] = round(total_holding_rounded, 2)
    df_cost.loc["TOTAL", "Total Cost (EUR)"]   = round(total_setup + total_holding_rounded, 2)

    grand_total = total_setup + total_holding_rounded + ot_total
    df_cost.loc["TOTAL", "Grand Total (EUR)"] = round(grand_total, 2)

    # Replace NaN with "" for cleaner display (part rows don't have OT costs)
    df_cost = df_cost.fillna("")

    print_cost_summary("Assignment 3a — Overtime (Forecasted Demand)", df_cost)

    # -------------------------------------------------------------------------
    # Overtime usage per period
    print("\nOvertime usage per period:")
    print(f"{'Period':<8} {'OT_X (units)':<16} {'OT_Y (min)':<14} {'OT_Y (hours)':<14}")
    for t in periods:
        print(f"  W{t:<5} {ot_x_vals[t]:<16.1f} {ot_y_vals[t]:<14.1f} {ot_y_vals[t]/60:.2f}")

    # -------------------------------------------------------------------------
    # Capacity utilisation summary
    CAP_X  = data["CAP_X"]
    CAP_Y  = data["CAP_Y"]
    PROC_Y = data["PROC_Y"]
    print("\nCapacity utilisation (base vs used vs available with OT):")
    print(f"{'Period':<8} {'WS-X prod':<12} {'WS-X cap':<12} {'WS-Y min used':<16} {'WS-Y cap':<12}")
    for t in periods:
        x_used  = p["E2801", t].X
        y_used  = PROC_Y["B1401"] * p["B1401", t].X + PROC_Y["B2302"] * p["B2302", t].X
        x_avail = CAP_X + ot_x_vals[t]
        y_avail = CAP_Y + ot_y_vals[t]
        print(f"  W{t:<5} {x_used:<12.0f} {x_avail:<12.0f} {y_used:<16.0f} {y_avail:<12.0f}")

    # -------------------------------------------------------------------------
    # Output DataFrames
    df_prod   = make_plan_df(p, parts, periods)
    df_inv    = make_plan_df(q, parts, periods)
    df_setup  = make_setup_df(y, parts, periods)
    df_demand = demand_row_df(demand, periods, label="Demand E2801 (Forecast)")

    ot_rows = []
    for t in periods:
        ot_rows.append({
            "Period":          f"W{t}",
            "OT WS-X (units)": round(ot_x_vals[t], 1),
            "OT WS-Y (min)":   round(ot_y_vals[t], 1),
            "OT WS-Y (hours)": round(ot_y_vals[t] / 60.0, 3),
        })
    df_overtime = pd.DataFrame(ot_rows).set_index("Period")

    _safe_write(
        "output_3a.xlsx",
        {
            "Production Plan": pd.concat([df_prod, df_demand]),
            "Inventory Plan":  df_inv,
            "Setup Plan":      df_setup,
            "Overtime Plan":   df_overtime,
            "Cost Summary":    df_cost,
        },
    )

    print(f"\nObjective value (Gurobi): {m.ObjVal:,.2f} EUR")
    print(f"Grand total (setup + holding + OT): {grand_total:,.2f} EUR")
    return m, grand_total


if __name__ == "__main__":
    data = load_data("input_data.json")
    solve_3a(data)