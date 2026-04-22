"""
Assignment 5a function
======================
Finite capacity with BOTH overtime and permanent modernization available.

The solver chooses the optimal combination of:
  - Permanent modernization (dx, dy): one-time investment
  - Weekly overtime (ox[t], oy[t]): per-period variable cost

Capacity constraint (combined):
  WS-X : p[E2801,t] <= CAP_X + dx + ox[t]
  WS-Y : 3*B1401 + 2*B2302 <= CAP_Y + (CAP_Y/100)*dy + oy[t]

5a : forecasted demand, no backorders

This file exposes solve_5a_plan(), so Assignment 5b can reuse
the exact fixed plan from 5a.
"""

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model,
    add_overtime_vars, add_modernization_vars,
    add_capacity_combined, set_combined_objective,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)


def solve_5a_plan(write_output=True, output_filename="output_5a.xlsx", print_summary=True):
    """
    Solve Assignment 5a and return the fixed plan.

    Returns a dictionary containing:
    - data, parts, periods, D_fcst
    - fixed production/setup/overtime/modernization decisions
    - reporting tables
    """

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    data           = load_data()
    parts          = data["parts"]
    periods        = data["periods"]
    demand         = data["D_fcst"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    m, p, q, y, _ = build_base_model(
        data, demand, "Assignment_5a", with_backorders=False
    )

    ox, oy = add_overtime_vars(m, data)
    dx, dy = add_modernization_vars(m, data)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, None, data, with_backorders=False)
    add_capacity_combined(m, p, ox, oy, dx, dy, data)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError("5a model not solved. Gurobi status: " + str(m.Status))

    # -------------------------------------------------------------------------
    # Cost components
    # -------------------------------------------------------------------------
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
        print_cost_summary("ASSIGNMENT 5A — Forecast Demand | Overtime + Modernization", df_cost)
        print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
        print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
        print("  Modernization WS-X : +" + "{:.1f}".format(dx.X) + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
        print("  Modernization WS-Y : +" + "{:.4f}".format(dy.X) + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
        print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
        print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
        print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))

    # -------------------------------------------------------------------------
    # Overtime schedule
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------------
    summary_rows = [
        {"Metric": "Setup Cost (EUR)",            "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",          "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR)",       "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR)",       "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)",  "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",  "Value": round(mod_cost_y, 2)},
        {"Metric": "Total Cost (EUR)",            "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)", "Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",     "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",   "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",     "Value": round(new_cap_y, 2)},
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

    # -------------------------------------------------------------------------
    # Fixed plan to reuse in 5b
    # -------------------------------------------------------------------------
    p_fix = {(i, t): int(round(p[i, t].X)) for i in parts for t in periods}
    y_fix = {(i, t): int(round(y[i, t].X)) for i in parts for t in periods}
    ox_fix = {t: float(ox[t].X) for t in periods}
    oy_fix = {t: float(oy[t].X) for t in periods}
    dx_fix = float(dx.X)
    dy_fix = float(dy.X)

    return {
        "data": data,
        "parts": parts,
        "periods": periods,
        "D_fcst": demand,
        "p_fix": p_fix,
        "y_fix": y_fix,
        "ox_fix": ox_fix,
        "oy_fix": oy_fix,
        "dx_fix": dx_fix,
        "dy_fix": dy_fix,
        "df_summary": df_summary,
        "df_cost": df_cost,
        "df_ot": df_ot,
        "df_prod": df_prod,
        "df_inv": df_inv,
        "df_setup": df_setup,
        "df_dem": df_dem,
    }


def main():
    solve_5a_plan(write_output=True, output_filename="output_5a.xlsx", print_summary=True)


if __name__ == "__main__":
    main()