"""
Assignment 6 — Extended Horizon (Phantom Week 31)
==================================================

APPROACH
--------
In assignment 5b, the fixed plan always drives inventory of E2801 to zero
by the end of week 30. This means that any positive demand surprise in the
final weeks immediately causes a backorder, because there is no buffer left.

The root cause: the optimizer knows the horizon ends at T=30, so it
rationally runs down inventory to zero. There is no incentive to hold stock
beyond week 30.

FIX: add a "phantom" week 31 with demand = average(W25..W30).
The optimizer now needs to keep enough inventory at the end of week 30 to
also cover (part of) week 31. This creates a natural end-of-horizon buffer
that absorbs late demand surges.

We use IDENTICAL model logic as 5a and 5b (same utils.py functions).
The only changes are:
  1. T is set to 31 internally.
  2. Demand list is extended by one value (avg W25..W30).
  3. Output reports only weeks 1..30 (week 31 phantom is excluded).
  4. Costs are computed only over weeks 1..30.
  5. For 6b: the fixed plan comes from 6a (this file's solve_6a_plan),
     evaluated under realized demand (also extended to 31 weeks with
     avg W25..W30 of the realized series).

Files produced:
  output_6a.xlsx  — 5a-equivalent, 31-week model, reporting weeks 1-30
  output_6b.xlsx  — 5b-equivalent evaluation, reporting weeks 1-30

Run:
  python 6.py
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


# =============================================================================
# HELPERS
# =============================================================================

def extend_demand(demand_30, label="forecast"):
    """
    Append week 31 = average(W25..W30) to a 30-week demand list.
    Returns a 31-element list and the W31 value.
    """
    avg_w31 = round(sum(demand_30[24:30]) / 6)
    print(f"  W31 ({label}): avg(W25..W30) = {sum(demand_30[24:30])/6:.2f} -> {avg_w31}")
    return demand_30 + [avg_w31], avg_w31


def patch_data_for_31_weeks(data):
    """
    Return a modified data dict with T=31 and both demand lists extended.
    The original data dict is NOT mutated.
    """
    import copy
    d = copy.deepcopy(data)
    d["T"]       = 31
    d["periods"] = list(range(1, 32))
    d["D_fcst"], _ = extend_demand(data["D_fcst"], "forecast")
    d["D_real"], _ = extend_demand(data["D_real"], "realized")
    return d


def slice_df_to_30(df):
    """Keep only columns W1..W30 from a wide DataFrame."""
    cols_30 = ["W" + str(t) for t in range(1, 31)]
    return df[[c for c in cols_30 if c in df.columns]]


def slice_cost_over_30(p, q, y, data30):
    """
    Compute setup + holding costs using only periods 1..30.
    Returns (rows_list, total_setup, total_holding).
    """
    parts   = data30["parts"]
    periods = data30["periods"]   # 1..30
    SC      = data30["SC"]
    HC      = data30["HC"]

    rows = []
    total_setup   = 0.0
    total_holding = 0.0

    for i in parts:
        s  = sum(SC[i] * round(y[i, t].X) for t in periods)
        h  = sum(HC[i] * q[i, t].X        for t in periods)
        ns = int(sum(round(y[i, t].X)      for t in periods))
        total_setup   += s
        total_holding += h
        rows.append({
            "Part":               i,
            "Num Setups":         ns,
            "Setup Cost (EUR)":   round(s, 2),
            "Holding Cost (EUR)": round(h, 2),
            "Total Cost (EUR)":   round(s + h, 2),
        })
    return rows, total_setup, total_holding


# =============================================================================
# 6a  — plan on FORECAST demand (31-week model, report weeks 1-30)
# =============================================================================

def solve_6a_plan(write_output=True,
                  output_filename="output_6a.xlsx",
                  print_summary=True):
    """
    Identical to 5a but with T=31 (phantom week 31 added).
    Only weeks 1-30 are reported in output and used for cost comparison.

    Returns the same dict structure as solve_5a_plan() so that solve_6b()
    can consume it without any changes.
    """

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    data30 = load_data()
    data31 = patch_data_for_31_weeks(data30)

    parts     = data31["parts"]
    periods   = data31["periods"]      # 1..31
    periods30 = data30["periods"]      # 1..30
    demand    = data31["D_fcst"]       # 31 values

    OT_COST_X      = data31["OT_COST_X"]
    OT_COST_Y      = data31["OT_COST_Y"]
    MOD_COST_X     = data31["MOD_COST_X"]
    MOD_COST_PCT_Y = data31["MOD_COST_PCT_Y"]
    CAP_X          = data31["CAP_X"]
    CAP_Y          = data31["CAP_Y"]

    # -------------------------------------------------------------------------
    # Model  (identical structure to 5a)
    # -------------------------------------------------------------------------
    m, p, q, y, _ = build_base_model(
        data31, demand, "Assignment_6a", with_backorders=False
    )

    ox, oy = add_overtime_vars(m, data31)
    dx, dy = add_modernization_vars(m, data31)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, None, data31,
                           with_backorders=False)
    add_capacity_combined(m, p, ox, oy, dx, dy, data31)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError("6a model not solved. Gurobi status: " + str(m.Status))

    # -------------------------------------------------------------------------
    # Cost components — weeks 1..30 ONLY
    # -------------------------------------------------------------------------
    total_ot_x   = sum(OT_COST_X * ox[t].X          for t in periods30)
    total_ot_y   = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods30)
    mod_cost_x   = MOD_COST_X * dx.X
    mod_cost_y   = MOD_COST_PCT_Y * dy.X
    new_cap_x    = CAP_X + dx.X
    new_cap_y    = CAP_Y * (1.0 + dy.X / 100.0)

    rows, total_setup, total_holding = slice_cost_over_30(p, q, y, data30)
    total_cost = (total_setup + total_holding
                  + total_ot_x + total_ot_y
                  + mod_cost_x + mod_cost_y)

    total_row = {
        "Part":                          "TOTAL",
        "Num Setups":                    sum(r["Num Setups"] for r in rows),
        "Setup Cost (EUR)":              round(total_setup, 2),
        "Holding Cost (EUR)":            round(total_holding, 2),
        "Total Cost (EUR)":              round(total_setup + total_holding, 2),
        "Overtime Cost X (EUR)":         round(total_ot_x, 2),
        "Overtime Cost Y (EUR)":         round(total_ot_y, 2),
        "Modernization Cost X (EUR)":    round(mod_cost_x, 2),
        "Modernization Cost Y (EUR)":    round(mod_cost_y, 2),
        "Grand Total (EUR)":             round(total_cost, 2),
    }
    rows.append(total_row)
    df_cost = pd.DataFrame(rows).set_index("Part")

    if print_summary:
        print_cost_summary(
            "ASSIGNMENT 6A — Forecast Demand | Extended Horizon (W31) | "
            "Costs reported W1-W30", df_cost)
        print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
        print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
        print("  Modernization WS-X : +" + "{:.1f}".format(dx.X)
              + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
        print("  Modernization WS-Y : +" + "{:.4f}".format(dy.X)
              + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
        print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
        print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
        print("  Total Cost (W1-30) : EUR " + "{:,.2f}".format(total_cost))

    # -------------------------------------------------------------------------
    # Overtime schedule — weeks 1..30 only
    # -------------------------------------------------------------------------
    ot_rows = []
    for t in periods30:
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
    # Summary
    # -------------------------------------------------------------------------
    summary_rows = [
        {"Metric": "Setup Cost (EUR)   [W1-W30]",    "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR) [W1-W30]",    "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR) [W1-W30]", "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR) [W1-W30]", "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)",      "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",      "Value": round(mod_cost_y, 2)},
        {"Metric": "Total Cost (EUR)   [W1-W30]",    "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)",     "Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",         "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",       "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",         "Value": round(new_cap_y, 2)},
        {"Metric": "W31 phantom demand (forecast)",   "Value": demand[30]},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    # Production / inventory / setup — sliced to weeks 1..30
    df_prod  = slice_df_to_30(make_plan_df(p, parts, periods))
    df_inv   = slice_df_to_30(make_plan_df(q, parts, periods))
    df_setup = slice_df_to_30(make_setup_df(y, parts, periods))
    df_dem   = demand_row_df(data30["D_fcst"], periods30,
                             label="Forecast Demand E2801")

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
    # Fixed plan to pass to 6b  (covers all 31 periods)
    # -------------------------------------------------------------------------
    p_fix  = {(i, t): int(round(p[i, t].X))  for i in parts for t in periods}
    y_fix  = {(i, t): int(round(y[i, t].X))  for i in parts for t in periods}
    ox_fix = {t: float(ox[t].X)              for t in periods}
    oy_fix = {t: float(oy[t].X)              for t in periods}
    dx_fix = float(dx.X)
    dy_fix = float(dy.X)

    return {
        "data":       data31,
        "data30":     data30,
        "parts":      parts,
        "periods":    periods,      # 1..31
        "periods30":  periods30,    # 1..30
        "D_fcst":     demand,
        "p_fix":      p_fix,
        "y_fix":      y_fix,
        "ox_fix":     ox_fix,
        "oy_fix":     oy_fix,
        "dx_fix":     dx_fix,
        "dy_fix":     dy_fix,
        "df_summary": df_summary,
        "df_cost":    df_cost,
        "df_ot":      df_ot,
        "df_prod":    df_prod,
        "df_inv":     df_inv,
        "df_setup":   df_setup,
        "df_dem":     df_dem,
    }


# =============================================================================
# 6b  — evaluate fixed 6a plan under REALIZED demand (report W1-W30)
# =============================================================================

def solve_6b(plan_6a=None,
             output_filename="output_6b.xlsx",
             print_summary=True):
    """
    Identical to 5b but uses the fixed plan from 6a (31-week model).
    Costs and service metrics are reported for weeks 1-30 only.
    Week 31 is the phantom week and excluded from all output.
    """

    if plan_6a is None:
        plan_6a = solve_6a_plan(write_output=False, print_summary=False)

    data31    = plan_6a["data"]
    data30    = plan_6a["data30"]
    parts     = plan_6a["parts"]
    periods   = plan_6a["periods"]      # 1..31
    periods30 = plan_6a["periods30"]    # 1..30

    p_fix   = plan_6a["p_fix"]
    y_fix   = plan_6a["y_fix"]
    ox_fix  = plan_6a["ox_fix"]
    oy_fix  = plan_6a["oy_fix"]
    dx_fix  = plan_6a["dx_fix"]
    dy_fix  = plan_6a["dy_fix"]

    D_fcst30 = data30["D_fcst"]   # 30 values — for reporting
    D_real31 = data31["D_real"]   # 31 values — for the model
    D_real30 = data30["D_real"]   # 30 values — for fill-rate denominator

    BO_COST        = data31["BO_COST"]
    OT_COST_X      = data31["OT_COST_X"]
    OT_COST_Y      = data31["OT_COST_Y"]
    MOD_COST_X     = data31["MOD_COST_X"]
    MOD_COST_PCT_Y = data31["MOD_COST_PCT_Y"]
    CAP_X          = data31["CAP_X"]
    CAP_Y          = data31["CAP_Y"]

    # -------------------------------------------------------------------------
    # Build 31-week model with backorders, realized demand
    # -------------------------------------------------------------------------
    m, p, q, y, b = build_base_model(
        data31, D_real31, "Assignment_6b", with_backorders=True
    )

    ox, oy = add_overtime_vars(m, data31)
    dx, dy = add_modernization_vars(m, data31)

    set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data31,
                           with_backorders=True)
    add_capacity_combined(m, p, ox, oy, dx, dy, data31)

    m.update()

    # Remove "no_final_backorder" — same reasoning as 5b
    c_final_bo = m.getConstrByName("no_final_backorder")
    if c_final_bo is not None:
        m.remove(c_final_bo)
        m.update()

    # Fix all 6a decisions (all 31 periods)
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
        raise RuntimeError("6b model not solved. Gurobi status: " + str(m.Status))

    # -------------------------------------------------------------------------
    # Costs — weeks 1..30 only
    # -------------------------------------------------------------------------
    total_ot_x   = sum(OT_COST_X * ox[t].X          for t in periods30)
    total_ot_y   = sum(OT_COST_Y * (oy[t].X / 60.0) for t in periods30)
    mod_cost_x   = MOD_COST_X * dx.X
    mod_cost_y   = MOD_COST_PCT_Y * dy.X
    total_bo     = sum(BO_COST * b[t].X              for t in periods30)
    new_cap_x    = CAP_X + dx.X
    new_cap_y    = CAP_Y * (1.0 + dy.X / 100.0)

    rows, total_setup, total_holding = slice_cost_over_30(p, q, y, data30)
    total_cost = (total_setup + total_holding
                  + total_ot_x + total_ot_y
                  + mod_cost_x + mod_cost_y
                  + total_bo)

    total_row = {
        "Part":                          "TOTAL",
        "Num Setups":                    sum(r["Num Setups"] for r in rows),
        "Setup Cost (EUR)":              round(total_setup, 2),
        "Holding Cost (EUR)":            round(total_holding, 2),
        "Total Cost (EUR)":              round(total_setup + total_holding, 2),
        "Overtime Cost X (EUR)":         round(total_ot_x, 2),
        "Overtime Cost Y (EUR)":         round(total_ot_y, 2),
        "Modernization Cost X (EUR)":    round(mod_cost_x, 2),
        "Modernization Cost Y (EUR)":    round(mod_cost_y, 2),
        "Backorder Cost (EUR)":          round(total_bo, 2),
        "Grand Total (EUR)":             round(total_cost, 2),
    }
    rows.append(total_row)
    df_cost = pd.DataFrame(rows).set_index("Part")

    # Service level & fill rate — weeks 1..30
    periods_with_bo = sum(1 for t in periods30 if b[t].X > 0.5)
    service_level   = 1.0 - periods_with_bo / len(periods30)

    new_backorders = sum(
        max(0.0, b[t].X - (b[t - 1].X if t > 1 else 0.0))
        for t in periods30
    )
    fill_rate = 1.0 - new_backorders / sum(D_real30)

    if print_summary:
        print_cost_summary(
            "ASSIGNMENT 6B — Fixed 6a Plan | Realized Demand | "
            "Extended Horizon (W31) | Costs reported W1-W30", df_cost)
        print("  Overtime Cost X    : EUR " + "{:,.2f}".format(total_ot_x))
        print("  Overtime Cost Y    : EUR " + "{:,.2f}".format(total_ot_y))
        print("  Modernization WS-X : +" + "{:.1f}".format(dx.X)
              + " units  -> EUR " + "{:,.2f}".format(mod_cost_x))
        print("  Modernization WS-Y : +" + "{:.4f}".format(dy.X)
              + "%      -> EUR " + "{:,.2f}".format(mod_cost_y))
        print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
        print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
        print("  Backorder Cost     : EUR " + "{:,.2f}".format(total_bo))
        print("  Total Cost (W1-30) : EUR " + "{:,.2f}".format(total_cost))
        print("  Service Level      :  " + "{:.2%}".format(service_level))
        print("  Fill Rate          :  " + "{:.2%}".format(fill_rate))

    # -------------------------------------------------------------------------
    # Output DataFrames — all sliced to weeks 1..30
    # -------------------------------------------------------------------------
    ot_rows = []
    for t in periods30:
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
        {"Metric": "Setup Cost (EUR)   [W1-W30]",    "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR) [W1-W30]",    "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR) [W1-W30]", "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR) [W1-W30]", "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)",      "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",      "Value": round(mod_cost_y, 2)},
        {"Metric": "Backorder Cost (EUR) [W1-W30]",  "Value": round(total_bo, 2)},
        {"Metric": "Total Cost (EUR)   [W1-W30]",    "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)",     "Value": round(dx.X, 2)},
        {"Metric": "Added capacity WS-Y (%)",         "Value": round(dy.X, 4)},
        {"Metric": "New capacity WS-X (units)",       "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",         "Value": round(new_cap_y, 2)},
        {"Metric": "Service Level      [W1-W30]",    "Value": round(service_level, 4)},
        {"Metric": "Fill Rate          [W1-W30]",    "Value": round(fill_rate, 4)},
        {"Metric": "W31 phantom demand (realized)",   "Value": D_real31[30]},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    df_prod  = slice_df_to_30(make_plan_df(p, parts, periods))
    df_inv   = slice_df_to_30(make_plan_df(q, parts, periods))
    df_setup = slice_df_to_30(make_setup_df(y, parts, periods))

    df_dem_fcst = demand_row_df(D_fcst30, periods30, label="Forecast Demand E2801")
    df_dem_real = demand_row_df(D_real30, periods30, label="Realized Demand E2801")

    bo_row = {"Part": "Backorder E2801"}
    for t in periods30:
        bo_row["W" + str(t)] = int(round(b[t].X))
    df_bo = pd.DataFrame([bo_row]).set_index("Part")

    write_excel(output_filename, {
        "Summary":         df_summary,
        "Cost per Part":   df_cost,
        "Overtime":        df_ot,
        "Production Plan": pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":  df_inv,
        "Setup Decisions": df_setup,
        "Backorders":      df_bo,
    })

    return {
        "total_cost":    total_cost,
        "service_level": service_level,
        "fill_rate":     fill_rate,
        "df_summary":    df_summary,
        "df_cost":       df_cost,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ASSIGNMENT 6 — Extended Horizon (Phantom Week 31)")
    print("  W31 forecast = avg(W25..W30) of forecast series")
    print("  W31 realized = avg(W25..W30) of realized series")
    print("  All output and costs are reported for weeks 1-30 only.")
    print("=" * 70)

    plan_6a = solve_6a_plan(
        write_output    = True,
        output_filename = "output_6a31W.xlsx",
        print_summary   = True,
    )

    print("\n" + "-" * 70)

    solve_6b(
        plan_6a         = plan_6a,
        output_filename = "output_6b31W.xlsx",
        print_summary   = True,
    )