"""
Assignment 3a & 3b
==================
Finite capacity with overtime option on both workstations.

WS-X overtime : EUR 2 / extra unit, max 300 units / period
WS-Y overtime : EUR 120 / hour (= EUR 2 / minute), max 38 hours / period

3a : forecasted demand, no backorders
3b : realized demand, backorders (EUR 250 / unit / period)

Output: output_3a.xlsx, output_3b.xlsx
"""

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model,
    add_overtime_vars, add_capacity_with_overtime, set_overtime_objective,
    compute_service_metrics,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)


def solve_3(demand, label, with_backorders):
    """Solve assignment 3 for the given demand scenario."""

    data      = load_data()
    parts     = data["parts"]
    periods   = data["periods"]
    BO_COST   = data["BO_COST"]
    OT_COST_X = data["OT_COST_X"]
    OT_COST_Y = data["OT_COST_Y"]

    # --- Build model ---
    m, p, q, y, b = build_base_model(
        data, demand, "Assignment_3_" + label, with_backorders=with_backorders
    )

    # Add overtime variables
    ox, oy = add_overtime_vars(m, data)

    # Replace base objective with full objective including overtime costs
    set_overtime_objective(m, p, q, y, ox, oy, b, data, with_backorders)

    # Add capacity constraints extended with overtime
    add_capacity_with_overtime(m, p, ox, oy, data)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("No solution for " + label + ". Status: " + str(m.Status))
        return

    # --- Costs ---
    total_ot_x = sum(OT_COST_X * ox[t].X for t in periods)
    total_ot_y = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods)

    extra = {
        "Overtime Cost X (EUR)": total_ot_x,
        "Overtime Cost Y (EUR)": total_ot_y,
    }
    if with_backorders:
        total_bo = sum(BO_COST * b[t].X for t in periods)
        extra["Backorder Cost (EUR)"] = total_bo

    df_cost, total_setup, total_holding = build_cost_summary(p, q, y, data, extra=extra)

    total_cost = total_setup + total_holding + total_ot_x + total_ot_y
    if with_backorders:
        total_cost += total_bo

    title = "ASSIGNMENT 3" + label.upper() + " — Finite Capacity + Overtime"
    print_cost_summary(title, df_cost)
    print("  Overtime Cost X : EUR " + "{:,.2f}".format(total_ot_x))
    print("  Overtime Cost Y : EUR " + "{:,.2f}".format(total_ot_y))
    print("  Total Cost      : EUR " + "{:,.2f}".format(total_cost))

    if with_backorders:
        service_level, fill_rate = compute_service_metrics(b, demand, periods)
        print("  Service Level   :  " + "{:.2%}".format(service_level))
        print("  Fill Rate       :  " + "{:.2%}".format(fill_rate))

    # --- Overtime schedule ---
    ot_rows = []
    for t in periods:
        ot_rows.append({
            "Period":            t,
            "OT Units X":        round(ox[t].X, 1),
            "OT Minutes Y":      round(oy[t].X, 1),
            "OT Hours Y":        round(oy[t].X / 60.0, 2),
            "OT Cost X (EUR)":   round(OT_COST_X * ox[t].X, 2),
            "OT Cost Y (EUR)":   round(OT_COST_Y * (oy[t].X / 60.0), 2),
        })
    df_ot = pd.DataFrame(ot_rows).set_index("Period")

    # --- Summary metrics ---
    summary_rows = [
        {"Metric": "Setup Cost (EUR)",      "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",    "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR)", "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR)", "Value": round(total_ot_y, 2)},
        {"Metric": "Total Cost (EUR)",      "Value": round(total_cost, 2)},
    ]
    if with_backorders:
        summary_rows.insert(4, {"Metric": "Backorder Cost (EUR)", "Value": round(total_bo, 2)})
        summary_rows.append({"Metric": "Service Level", "Value": round(service_level, 4)})
        summary_rows.append({"Metric": "Fill Rate",     "Value": round(fill_rate, 4)})
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)
    d_label  = "Realized Demand E2801" if with_backorders else "Forecast Demand E2801"
    df_dem   = demand_row_df(demand, periods, label=d_label)

    fname = "output_3a.xlsx" if not with_backorders else "output_3b.xlsx"
    write_excel(fname, {
        "Summary":         df_summary,
        "Cost per Part":   df_cost,
        "Overtime":        df_ot,
        "Production Plan": pd.concat([df_prod, df_dem]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
    })


# -----------------------------------------------------------------------------
# Run both scenarios
# -----------------------------------------------------------------------------
data = load_data()
solve_3(data["D_fcst"], "3a", with_backorders=False)
solve_3(data["D_real"], "3b", with_backorders=True)