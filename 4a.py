"""
Assignment 4a & 4b
==================
Finite capacity with permanent modernization option.
 
WS-X modernization : EUR 10 / extra unit of permanent capacity, max 200 units
WS-Y modernization : EUR 1500 / 1% capacity increase, in increments of EUR 15
                     (= 0.01% per increment), max 40%
 
The investment is paid ONCE and applies to all 30 periods.
 
4a : forecasted demand, no backorders
4b : realized demand, backorders (EUR 250 / unit / period)
 
Output: output_4a.xlsx, output_4b.xlsx
"""
 
import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model,
    add_modernization_vars, add_capacity_with_modernization,
    set_modernization_objective,
    compute_service_metrics,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)
 
 
def solve_4(demand, label, with_backorders):
    """Solve assignment 4 for the given demand scenario."""
 
    data           = load_data()
    parts          = data["parts"]
    periods        = data["periods"]
    BO_COST        = data["BO_COST"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]
 
    # --- Build model ---
    m, p, q, y, b = build_base_model(
        data, demand, "Assignment_4_" + label, with_backorders=with_backorders
    )
 
    # Add permanent modernization variables
    dx, dy = add_modernization_vars(m, data)
 
    # Set full objective including modernization investment cost
    set_modernization_objective(m, p, q, y, dx, dy, b, data, with_backorders)
 
    # Add capacity constraints extended with permanent modernization
    add_capacity_with_modernization(m, p, dx, dy, data)
 
    m.optimize()
 
    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("No solution for " + label + ". Status: " + str(m.Status))
        return
 
    # --- Modernization results ---
    mod_cost_x = MOD_COST_X * dx.X
    mod_cost_y = MOD_COST_PCT_Y * dy.X
    new_cap_x  = CAP_X + dx.X
    new_cap_y  = CAP_Y * (1.0 + dy.X / 100.0)
 
    extra = {
        "Modernization Cost X (EUR)": mod_cost_x,
        "Modernization Cost Y (EUR)": mod_cost_y,
    }
    if with_backorders:
        total_bo = sum(BO_COST * b[t].X for t in periods)
        extra["Backorder Cost (EUR)"] = total_bo
 
    df_cost, total_setup, total_holding = build_cost_summary(p, q, y, data, extra=extra)
    total_cost = total_setup + total_holding + mod_cost_x + mod_cost_y
    if with_backorders:
        total_cost += total_bo
 
    title = "ASSIGNMENT " + label.upper() + " — Finite Capacity + Modernization"
    print_cost_summary(title, df_cost)
    print("  Modernization WS-X : +"  + "{:.1f}".format(dx.X) + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
    print("  Modernization WS-Y : +"  + "{:.4f}".format(dy.X) + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
    print("  New capacity WS-X  :  "  + "{:.1f}".format(new_cap_x) + " units/week")
    print("  New capacity WS-Y  :  "  + "{:.1f}".format(new_cap_y) + " min/week")
    print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))
 
    if with_backorders:
        service_level, fill_rate = compute_service_metrics(b, demand, periods)
        print("  Service Level   :  " + "{:.2%}".format(service_level))
        print("  Fill Rate       :  " + "{:.2%}".format(fill_rate))
 
    # --- Summary metrics ---
    summary_rows = [
        {"Metric": "Setup Cost (EUR)",            "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",          "Value": round(total_holding, 2)},
        {"Metric": "Modernization Cost X (EUR)",  "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",  "Value": round(mod_cost_y, 2)},
        {"Metric": "Total Cost (EUR)",            "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)", "Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",     "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",   "Value": round(new_cap_x, 1)},
        {"Metric": "New capacity WS-Y (min/wk)",  "Value": round(new_cap_y, 1)},
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
 
    fname = "output_4a.xlsx" if not with_backorders else "output_4b.xlsx"
    write_excel(fname, {
        "Summary":         df_summary,
        "Cost per Part":   df_cost,
        "Production Plan": pd.concat([df_prod, df_dem]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
    })
 
 
# -----------------------------------------------------------------------------
# Run both scenarios
# -----------------------------------------------------------------------------
data = load_data()
solve_4(data["D_fcst"], "4a", with_backorders=False)
solve_4(data["D_real"], "4b", with_backorders=True)