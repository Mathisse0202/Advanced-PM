"""
Assignment 5b
=============
Takes the FIXED production plan from 5a (optimized on forecasted demand)
and evaluates it against realized demand.
Backorders occur where the fixed plan cannot meet realized demand (EUR 250/unit/period).

Output: output_5b.xlsx
"""

import pandas as pd
from gurobipy import GRB
from utils import (
    load_data, build_base_model,
    add_overtime_vars, add_modernization_vars,
    add_capacity_combined, set_combined_objective,
    make_plan_df, make_setup_df, demand_row_df,
    write_excel, print_cost_summary,
)


def simulate_5b(p, q, ox, oy, dx, dy, data):
    """
    Given the fixed 5a production plan variables, simulate period-by-period
    inventory and backorders for E2801 using REALIZED demand.

    Inventory balance per period for E2801:
        arriving[t]  = p[E2801, t - LT["E2801"]]   (production ordered LT periods ago)
        net[t]       = q_prev + b_prev + arriving[t] - D_real[t]
        if net >= 0  ->  q_new[t] = net,  b[t] = 0
        if net <  0  ->  q_new[t] = 0,    b[t] = |net|

    All other parts are unaffected — their production and inventory are fixed
    by the 5a plan since internal demand (driven by E2801 production) is unchanged.
    """
    periods  = data["periods"]
    D_real   = data["D_real"]
    I0       = data["I0"]
    LT       = data["LT"]

    inv_E2801  = {}   # simulated inventory for E2801
    backorders = {}   # backorder quantity per period

    q_prev = I0["E2801"]
    b_prev = 0.0

    for t in periods:
        # Production that arrives in period t (ordered LT periods ago)
        t_order  = t - LT["E2801"]
        arriving = p["E2801", t_order].X if t_order >= 1 else 0.0

        demand_t = D_real[t - 1]   # D_real is 0-indexed

        net = q_prev + b_prev + arriving - demand_t

        if net >= 0:
            inv_E2801[t]  = net
            backorders[t] = 0.0
        else:
            inv_E2801[t]  = 0.0
            backorders[t] = -net

        q_prev = inv_E2801[t]
        b_prev = backorders[t]

    return inv_E2801, backorders


def solve_5b():
    data           = load_data()
    parts          = data["parts"]
    periods        = data["periods"]
    SC             = data["SC"]
    HC             = data["HC"]
    BO_COST        = data["BO_COST"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]

    # ---------------------------------------------------------------
    # Step 1: Re-solve 5a with FORECASTED demand to get the fixed plan
    # ---------------------------------------------------------------
    print("\nSolving 5a plan (forecasted demand)...")
    m, p, q, y, b = build_base_model(
        data, data["D_fcst"], "Assignment_5b_base", with_backorders=False
    )

    ox, oy = add_overtime_vars(m, data)
    dx, dy = add_modernization_vars(m, data)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data, with_backorders=False)
    add_capacity_combined(m, p, ox, oy, dx, dy, data)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("Could not obtain 5a plan. Status: " + str(m.Status))
        return

    # ---------------------------------------------------------------
    # Step 2: Simulate realized demand -> compute E2801 backorders
    # ---------------------------------------------------------------
    print("\nSimulating realized demand against fixed plan...")
    inv_E2801, backorders = simulate_5b(p, q, ox, oy, dx, dy, data)

    # ---------------------------------------------------------------
    # Step 3: Recompute all costs
    #   - Setup, OT, modernization: identical to 5a (plan is fixed)
    #   - Holding for non-E2801: identical to 5a (internal demand unchanged)
    #   - Holding for E2801: use simulated inventory (not 5a q values)
    #   - Backorder: from simulation
    # ---------------------------------------------------------------
    total_ot_x = sum(OT_COST_X *  ox[t].X          for t in periods)
    total_ot_y = sum(OT_COST_Y * (oy[t].X / 60.0)  for t in periods)
    mod_cost_x = MOD_COST_X     * dx.X
    mod_cost_y = MOD_COST_PCT_Y * dy.X
    total_bo   = sum(BO_COST    * backorders[t]      for t in periods)
    new_cap_x  = CAP_X + dx.X
    new_cap_y  = CAP_Y * (1.0 + dy.X / 100.0)

    # Setup cost: same y values as 5a
    total_setup = sum(SC[i] * y[i, t].X for i in parts for t in periods)

    # Holding cost: all parts EXCEPT E2801 use q from 5a
    total_holding = 0.0
    for i in parts:
        if i == "E2801":
            # Use simulated inventory driven by realized demand
            total_holding += sum(HC["E2801"] * inv_E2801[t] for t in periods)
        else:
            # Internal demand unchanged — 5a q values still valid
            total_holding += sum(HC[i] * q[i, t].X for t in periods)

    total_cost = (total_setup + total_holding
                  + total_ot_x + total_ot_y
                  + mod_cost_x + mod_cost_y
                  + total_bo)

    # ---------------------------------------------------------------
    # Step 4: Service metrics
    # ---------------------------------------------------------------
    D_real        = data["D_real"]
    n_periods     = len(periods)
    service_level = sum(1 for t in periods if backorders[t] == 0) / n_periods
    total_demand  = sum(D_real)
    fill_rate     = 1.0 - sum(backorders[t] for t in periods) / total_demand

    # ---------------------------------------------------------------
    # Step 5: Print results
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ASSIGNMENT 5B — Fixed 5a plan evaluated on realized demand")
    print("=" * 70)
    print("  Setup Cost         : EUR " + "{:,.2f}".format(total_setup))
    print("  Holding Cost       : EUR " + "{:,.2f}".format(total_holding))
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
    print("=" * 70)

    print("\n--- Output Summary ---")
    print({
        "total_cost": round(total_cost, 1),
        "cost_breakdown": {
            "setup_cost":        round(total_setup, 1),
            "holding_cost":      round(total_holding, 1),
            "backorder_cost":    round(total_bo, 1),
            "investment_cost_X": round(mod_cost_x, 1),
            "investment_cost_Y": round(mod_cost_y, 1),
            "overtime_cost_X":   round(total_ot_x, 1),
            "overtime_cost_Y":   round(total_ot_y, 1),
        }
    })

    # ---------------------------------------------------------------
    # Step 6: Build output DataFrames and write Excel
    # ---------------------------------------------------------------
    # Overtime schedule
    ot_rows = [{
        "Period":          t,
        "OT Units X":      round(ox[t].X, 1),
        "OT Minutes Y":    round(oy[t].X, 1),
        "OT Hours Y":      round(oy[t].X / 60.0, 2),
        "OT Cost X (EUR)": round(OT_COST_X * ox[t].X, 2),
        "OT Cost Y (EUR)": round(OT_COST_Y * (oy[t].X / 60.0), 2),
    } for t in periods]
    df_ot = pd.DataFrame(ot_rows).set_index("Period")

    # Backorder schedule
    bo_rows = [{
        "Period":              t,
        "Backorder (units)":   round(backorders[t], 1),
        "Backorder Cost (EUR)": round(BO_COST * backorders[t], 2),
    } for t in periods]
    df_bo = pd.DataFrame(bo_rows).set_index("Period")

    # Summary sheet
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

    # Production/setup plans — pass Gurobi vars directly (make_plan_df calls .X internally)
    df_prod  = make_plan_df(p, parts, periods)
    df_inv   = make_plan_df(q, parts, periods)
    df_setup = make_setup_df(y, parts, periods)

    # Append both demand rows so forecast vs realized is visible side-by-side
    df_dem_fcst = demand_row_df(data["D_fcst"], periods, label="Forecast Demand E2801")
    df_dem_real = demand_row_df(data["D_real"], periods, label="Realized Demand E2801")

    write_excel("output_5b.xlsx", {
        "Summary":         df_summary,
        "Overtime":        df_ot,
        "Backorders":      df_bo,
        "Production Plan": pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
    })


solve_5b()