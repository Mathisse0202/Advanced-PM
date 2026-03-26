"""
Assignment 5a & 5b
==================
Finite capacity with BOTH overtime and permanent modernization available.

The solver chooses the optimal combination of:
  - Permanent modernization (dx, dy) : one-time investment
  - Weekly overtime (ox[t], oy[t])   : per-period variable cost

Capacity constraint (combined):
  WS-X : p[E2801,t] <= CAP_X + dx + ox[t]
  WS-Y : 3*B1401 + 2*B2302 <= CAP_Y + (CAP_Y/100)*dy + oy[t]

5a : forecasted demand, no backorders
5b : realized demand, backorders (EUR 250 / unit / period)

Output: output_5a.xlsx, output_5b.xlsx
"""

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model,
    add_overtime_vars, add_modernization_vars,
    add_capacity_combined, set_combined_objective,
    compute_service_metrics,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)


def solve_5(demand, label, with_backorders):
    """Solve assignment 5 for the given demand scenario."""

    data           = load_data()
    parts          = data["parts"]
    periods        = data["periods"]
    BO_COST        = data["BO_COST"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]

    # --- Build model ---
    m, p, q, y, b = build_base_model(
        data, demand, "Assignment_5_" + label, with_backorders=with_backorders
    )

    # Add both capacity expansion options
    ox, oy = add_overtime_vars(m, data)
    dx, dy = add_modernization_vars(m, data)

    # Full objective: setup + holding + overtime + modernization + (backorder)
    set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data, with_backorders)

    # Combined capacity constraints
    add_capacity_combined(m, p, ox, oy, dx, dy, data)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("No solution for " + label + ". Status: " + str(m.Status))
        return

    # --- Cost components ---
    total_ot_x   = sum(OT_COST_X * ox[t].X for t in periods)
    total_ot_y   = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods)
    mod_cost_x   = MOD_COST_X * dx.X
    mod_cost_y   = MOD_COST_PCT_Y * dy.X
    new_cap_x    = CAP_X + dx.X
    new_cap_y    = CAP_Y * (1.0 + dy.X / 100.0)

    extra = {
        "Overtime Cost X (EUR)":       total_ot_x,
        "Overtime Cost Y (EUR)":       total_ot_y,
        "Modernization Cost X (EUR)":  mod_cost_x,
        "Modernization Cost Y (EUR)":  mod_cost_y,
    }
    if with_backorders:
        total_bo = sum(BO_COST * b[t].X for t in periods)
        extra["Backorder Cost (EUR)"] = total_bo

    df_cost, total_setup, total_holding = build_cost_summary(p, q, y, data, extra=extra)
    total_cost = total_setup + total_holding + total_ot_x + total_ot_y + mod_cost_x + mod_cost_y
    if with_backorders:
        total_cost += total_bo

    title = "ASSIGNMENT " + label.upper() + " — Finite Capacity + Overtime + Modernization"
    print_cost_summary(title, df_cost)
    print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
    print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
    print("  Modernization WS-X : +"   + "{:.1f}".format(dx.X) + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
    print("  Modernization WS-Y : +"   + "{:.4f}".format(dy.X) + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
    print("  New capacity WS-X  :  "   + "{:.1f}".format(new_cap_x) + " units/week")
    print("  New capacity WS-Y  :  "   + "{:.1f}".format(new_cap_y) + " min/week")
    print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))

    if with_backorders:
        service_level, fill_rate = compute_service_metrics(b, demand, periods)
        print("  Service Level   :  " + "{:.2%}".format(service_level))
        print("  Fill Rate       :  " + "{:.2%}".format(fill_rate))

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

    # --- Summary metrics ---
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
    ]
    if with_backorders:
        summary_rows.insert(6, {"Metric": "Backorder Cost (EUR)", "Value": round(total_bo, 2)})
        summary_rows.append({"Metric": "Service Level", "Value": round(service_level, 4)})
        summary_rows.append({"Metric": "Fill Rate",     "Value": round(fill_rate, 4)})
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)
    d_label  = "Realized Demand E2801" if with_backorders else "Forecast Demand E2801"
    df_dem   = demand_row_df(demand, periods, label=d_label)

    fname = "output_5a.xlsx" if not with_backorders else "output_5b.xlsx"
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
solve_5(data["D_fcst"], "5a", with_backorders=False)
solve_5(data["D_real"], "5b", with_backorders=True)