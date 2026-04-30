"""
Assignment 6
============
Handling demand uncertainty: Rolling Horizon with Safety Stock.

APPROACH RATIONALE
------------------
In 5b, the fixed 5a plan (optimized on forecasts) causes backorders in
weeks 26 and 30, with a total backorder cost of EUR 37,250. The root
cause is that realized demand systematically exceeds forecasts over the
30-week horizon (total surplus: +110 units), and the plan has no
mechanism to react once production decisions are locked in.

Two improvements are combined:

1. ROLLING HORIZON (RH)
   Instead of solving once for all 30 weeks and freezing the plan, we
   re-optimize every R periods using the most recent demand information.
   Only the first R periods of each solution are executed (frozen);
   then we re-solve with updated data.
   - The "look-ahead" window W is kept at 30 - t + 1 (full remaining
     horizon), so the optimizer always sees the long-term consequences.
   - Re-solving allows the plan to absorb forecast errors period by period.

2. SAFETY STOCK (SS) for E2801
   Even within a rolling horizon there is residual uncertainty because
   we still plan against forecasts for future periods. Adding a minimum
   inventory buffer (safety stock) for the end product guarantees that
   small positive demand surprises are absorbed from stock rather than
   becoming backorders.

   SS = z * sigma_e * sqrt(LT_E2801 + 1)
      = 1.645 * 24.9 * sqrt(2)  ≈  58 units   (95% service level target)

   This is implemented as a lower bound on q[E2801, t] for each period.

WHY THESE TWO AND NOT OTHERS?
- Safety stock alone cannot fix large consecutive surprises (W26 cumulative
  drift); the rolling horizon handles that.
- Rolling horizon alone with no buffer still leaves the plan vulnerable
  to within-window surprises (SS fixes that).
- Stochastic programming / scenario trees would be theoretically superior
  but require probability distributions that are not given; the empirical
  error std (24.9 units) is sufficient for a safety-stock calculation.
- The combination is implementable within the existing Gurobi framework
  with minimal changes.

PARAMETERS (can be tuned)
--------------------------
  RH_FREQ  : re-planning frequency (periods between re-solves); default = 5
  SS_E2801 : safety stock for E2801; default = 58 units (95% CSL)

OUTPUT
------
  output_6.xlsx  — same sheet structure as 5b, plus a "Method Summary" sheet.
"""

import math
import numpy as np
import pandas as pd
from gurobipy import GRB
from utils import (
    load_data,
    add_overtime_vars, add_modernization_vars,
    add_capacity_combined, set_combined_objective,
    compute_service_metrics,
    make_plan_df, make_setup_df, build_cost_summary, demand_row_df,
    write_excel, print_cost_summary,
)
import gurobipy as gp


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
RH_FREQ  = 5    # Re-plan every RH_FREQ periods
SS_E2801 = 58   # Safety stock for E2801 (units); based on 95% CSL


# =============================================================================
# HELPER: solve one rolling-horizon window
# =============================================================================

def solve_window(data, demand_window, I0_window, T_window,
                 dx_fix, dy_fix, ox_fix_dict, oy_fix_dict,
                 frozen_p, frozen_y, window_start,
                 with_ss=True):
    """
    Solve a single planning window.

    Parameters
    ----------
    data          : full data dict from load_data()
    demand_window : list of length T_window — forecast demand for this window
    I0_window     : dict {part: starting inventory} at the start of this window
    T_window      : number of periods in this window
    dx_fix, dy_fix: fixed modernization from 5a
    ox_fix_dict   : {abs_period: ox_value} already fixed (periods before window)
    oy_fix_dict   : {abs_period: oy_value} already fixed
    frozen_p      : {(part, abs_period): qty} decisions already executed
    frozen_y      : {(part, abs_period): 0/1} decisions already executed
    window_start  : 1-based absolute period of the first period in this window
    with_ss       : whether to enforce safety stock constraint on E2801

    Returns
    -------
    p_sol, y_sol, ox_sol, oy_sol : dicts keyed by local period (1..T_window)
    obj_val : objective value of this window's model
    """
    parts   = data["parts"]
    LT      = data["LT"]
    Q_min   = data["Q_min"]
    BOM     = data["BOM"]
    parents = data["parents"]  # child -> {parent: qty}
    SC      = data["SC"]
    HC      = data["HC"]
    BIG_M   = data["BIG_M"]
    BO_COST = data["BO_COST"]
    CAP_X   = data["CAP_X"]
    CAP_Y   = data["CAP_Y"]
    PROC_Y  = data["PROC_Y"]
    OT_COST_X = data["OT_COST_X"]
    OT_COST_Y = data["OT_COST_Y"]
    OT_MAX_X  = data["OT_MAX_X"]
    OT_MAX_Y  = data["OT_MAX_Y"]

    periods_local = list(range(1, T_window + 1))

    m = gp.Model("window_" + str(window_start))
    m.setParam("OutputFlag", 0)
    m.setParam("MIPGap", 1e-4)

    # Decision variables (local period indexing)
    p  = m.addVars(parts, periods_local, name="p", lb=0.0, vtype=GRB.INTEGER)
    q  = m.addVars(parts, periods_local, name="q", lb=0.0, vtype=GRB.INTEGER)
    y  = m.addVars(parts, periods_local, name="y", vtype=GRB.BINARY)
    ox = m.addVars(periods_local, name="ox", lb=0.0, ub=OT_MAX_X, vtype=GRB.INTEGER)
    oy = m.addVars(periods_local, name="oy", lb=0.0, ub=OT_MAX_Y)

    # --- Objective ---
    obj = (
        gp.quicksum(SC[i] * y[i, t] + HC[i] * q[i, t]
                    for i in parts for t in periods_local)
        + gp.quicksum(OT_COST_X * ox[t] for t in periods_local)
        + gp.quicksum(OT_COST_Y * (oy[t] / 60.0) for t in periods_local)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # --- Inventory balance and lot sizing ---
    from collections import defaultdict

    for i in parts:
        for t_loc in periods_local:
            t_abs = window_start + t_loc - 1

            q_prev = I0_window[i] if t_loc == 1 else q[i, t_loc - 1]

            # Production ordered t_abs - LT[i] must have arrived
            t_order_abs = t_abs - LT[i]
            t_order_loc = t_order_abs - window_start + 1

            if t_order_abs < 1:
                arriving = 0.0
            elif t_order_loc < 1:
                # Was ordered before this window => already executed
                arriving = frozen_p.get((i, t_order_abs), 0)
            else:
                arriving = p[i, t_order_loc]

            ext_demand = demand_window[t_loc - 1] if i == "E2801" else 0
            ind_demand = (
                gp.quicksum(BOM[j][i] * p[j, t_loc] for j in parents[i]
                            if j in BOM and i in BOM[j])
                if i in parents else 0
            )

            m.addConstr(
                q_prev + arriving == ext_demand + ind_demand + q[i, t_loc],
                name="inv_" + i + "_" + str(t_loc)
            )
            m.addConstr(p[i, t_loc] >= Q_min[i] * y[i, t_loc],
                        name="minlot_" + i + "_" + str(t_loc))
            m.addConstr(p[i, t_loc] <= BIG_M * y[i, t_loc],
                        name="force_" + i + "_" + str(t_loc))

            # Safety stock: inventory of E2801 must stay >= SS_E2801
            if with_ss and i == "E2801":
                m.addConstr(q[i, t_loc] >= SS_E2801,
                            name="ss_E2801_" + str(t_loc))

    # --- Capacity with overtime + fixed modernization ---
    for t_loc in periods_local:
        m.addConstr(
            p["E2801", t_loc] <= CAP_X + dx_fix + ox[t_loc],
            name="cap_X_" + str(t_loc)
        )
        m.addConstr(
            PROC_Y["B1401"] * p["B1401", t_loc]
            + PROC_Y["B2302"] * p["B2302", t_loc]
            <= CAP_Y + (CAP_Y / 100.0) * dy_fix + oy[t_loc],
            name="cap_Y_" + str(t_loc)
        )

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(
            f"Window starting at {window_start} infeasible. "
            f"Gurobi status: {m.Status}"
        )

    p_sol  = {t: {i: int(round(p[i, t].X))  for i in parts} for t in periods_local}
    y_sol  = {t: {i: int(round(y[i, t].X))  for i in parts} for t in periods_local}
    ox_sol = {t: float(ox[t].X) for t in periods_local}
    oy_sol = {t: float(oy[t].X) for t in periods_local}

    return p_sol, y_sol, ox_sol, oy_sol


# =============================================================================
# MAIN: Rolling Horizon simulation
# =============================================================================

def solve_6(rh_freq=RH_FREQ, ss=SS_E2801,
            output_filename="output_6.xlsx", print_summary=True):

    data    = load_data()
    parts   = data["parts"]
    T       = data["T"]
    periods = data["periods"]
    D_fcst  = data["D_fcst"]
    D_real  = data["D_real"]

    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    CAP_X          = data["CAP_X"]
    CAP_Y          = data["CAP_Y"]

    # ------------------------------------------------------------------
    # Step 1: Solve 5a once to get the optimal modernization investment
    # (dx, dy). Overtime decisions will be re-optimized each window.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ASSIGNMENT 6 — Rolling Horizon + Safety Stock")
    print("  Step 1: Determining optimal modernization via 5a model...")
    print("=" * 70)

    import importlib.util
    from pathlib import Path
    module_path = Path(__file__).with_name("5aFUNCTION.py")
    spec = importlib.util.spec_from_file_location("assignment5a_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plan_5a = module.solve_5a_plan(write_output=False, print_summary=False)
    dx_fix  = plan_5a["dx_fix"]
    dy_fix  = plan_5a["dy_fix"]

    print(f"  Modernization fixed: WS-X +{dx_fix:.0f} units, WS-Y +{dy_fix:.4f}%")
    print(f"  (Modernization cost: EUR {MOD_COST_X*dx_fix + MOD_COST_PCT_Y*dy_fix:,.2f})")

    # ------------------------------------------------------------------
    # Step 2: Rolling horizon simulation using REALIZED demand to
    # evaluate, but FORECAST demand to plan future periods.
    # ------------------------------------------------------------------
    print(f"\n  Step 2: Rolling Horizon (replan every {rh_freq} period(s), "
          f"SS={ss} units)...")

    # Tracking structures
    p_executed  = {}   # (part, abs_period) -> qty actually produced
    y_executed  = {}   # (part, abs_period) -> 0/1
    ox_executed = {}   # abs_period -> float
    oy_executed = {}   # abs_period -> float
    inv_real    = {}   # (part, abs_period) -> inventory under realized demand
    bo_real     = {}   # abs_period -> backorder of E2801

    # Starting inventory
    I0_current = {i: data["I0"][i] for i in parts}

    t = 1
    while t <= T:
        # Determine the window to plan
        window_start = t
        T_window     = T - t + 1   # look ahead to end of horizon

        # Forecast for this window (future is still uncertain -> use D_fcst)
        demand_window = D_fcst[t - 1:]

        if print_summary:
            print(f"    Re-planning from period {window_start} "
                  f"(window={T_window} periods)...")

        p_sol, y_sol, ox_sol, oy_sol = solve_window(
            data           = data,
            demand_window  = demand_window,
            I0_window      = I0_current,
            T_window       = T_window,
            dx_fix         = dx_fix,
            dy_fix         = dy_fix,
            ox_fix_dict    = ox_executed,
            oy_fix_dict    = oy_executed,
            frozen_p       = p_executed,
            frozen_y       = y_executed,
            window_start   = window_start,
            with_ss        = (ss > 0),
        )

        # Execute the first rh_freq periods (or fewer at end of horizon)
        n_execute = min(rh_freq, T_window)

        for t_loc in range(1, n_execute + 1):
            t_abs = window_start + t_loc - 1

            # Record production / setup / overtime decisions
            for i in parts:
                p_executed[i, t_abs] = p_sol[t_loc][i]
                y_executed[i, t_abs] = y_sol[t_loc][i]
            ox_executed[t_abs] = ox_sol[t_loc]
            oy_executed[t_abs] = oy_sol[t_loc]

        # Now compute ACTUAL inventory under realized demand for the
        # executed periods.  We need to walk through period by period
        # because inventory depends on what actually arrived (based on
        # lead times and executed production).
        for t_loc in range(1, n_execute + 1):
            t_abs = window_start + t_loc - 1

            for i in parts:
                # Inventory carried from previous period (or t=0)
                if t_abs == 1:
                    q_prev_r = data["I0"][i]
                else:
                    q_prev_r = inv_real.get((i, t_abs - 1), 0)

                # Production ordered LT[i] periods ago arrives now
                t_order_abs = t_abs - data["LT"][i]
                arriving = p_executed.get((i, t_order_abs), 0) if t_order_abs >= 1 else 0

                # Demand
                ext_demand = D_real[t_abs - 1] if i == "E2801" else 0
                ind_demand = sum(
                    data["BOM"][j][i] * p_executed.get((j, t_abs), 0)
                    for j in data["parents"].get(i, {})
                    if j in data["BOM"] and i in data["BOM"][j]
                )

                if i == "E2801":
                    bo_prev = bo_real.get(t_abs - 1, 0) if t_abs > 1 else 0
                    net     = q_prev_r + arriving + bo_prev - ext_demand
                    if net >= 0:
                        inv_real[i, t_abs] = net
                        bo_real[t_abs]     = 0
                    else:
                        inv_real[i, t_abs] = 0
                        bo_real[t_abs]     = -net
                else:
                    inv_real[i, t_abs] = max(0, q_prev_r + arriving - ext_demand - ind_demand)

        # Update starting inventory for next re-plan
        last_exec = window_start + n_execute - 1
        for i in parts:
            I0_current[i] = inv_real.get((i, last_exec), 0)

        t += n_execute

    # ------------------------------------------------------------------
    # Step 3: Compute costs under realized demand
    # ------------------------------------------------------------------
    mod_cost_x = MOD_COST_X * dx_fix
    mod_cost_y = MOD_COST_PCT_Y * dy_fix

    total_setup   = sum(
        data["SC"][i] * y_executed.get((i, t), 0)
        for i in parts for t in periods
    )
    total_holding = sum(
        data["HC"][i] * inv_real.get((i, t), 0)
        for i in parts for t in periods
    )
    total_ot_x = sum(
        OT_COST_X * ox_executed.get(t, 0) for t in periods
    )
    total_ot_y = sum(
        OT_COST_Y * (oy_executed.get(t, 0) / 60.0) for t in periods
    )
    total_bo = sum(
        data["BO_COST"] * bo_real.get(t, 0) for t in periods
    )

    total_cost = (
        total_setup + total_holding
        + total_ot_x + total_ot_y
        + mod_cost_x + mod_cost_y
        + total_bo
    )

    # Service level & fill rate
    bo_vals = {t: bo_real.get(t, 0) for t in periods}
    periods_with_bo = sum(1 for t in periods if bo_vals[t] > 0.5)
    service_level   = 1.0 - periods_with_bo / len(periods)

    new_backorders = sum(
        max(0, bo_vals[t] - (bo_vals[t-1] if t > 1 else 0))
        for t in periods
    )
    fill_rate = 1.0 - new_backorders / sum(D_real)

    new_cap_x = CAP_X + dx_fix
    new_cap_y = CAP_Y * (1.0 + dy_fix / 100.0)

    # ------------------------------------------------------------------
    # Step 4: Build output DataFrames
    # ------------------------------------------------------------------

    # Cost per part
    rows = []
    for i in parts:
        s  = sum(data["SC"][i] * y_executed.get((i, t), 0) for t in periods)
        h  = sum(data["HC"][i] * inv_real.get((i, t), 0) for t in periods)
        ns = int(sum(y_executed.get((i, t), 0) for t in periods))
        rows.append({
            "Part":               i,
            "Num Setups":         ns,
            "Setup Cost (EUR)":   round(s, 2),
            "Holding Cost (EUR)": round(h, 2),
            "Total Cost (EUR)":   round(s + h, 2),
        })
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

    # Summary
    summary_rows = [
        {"Metric": "Setup Cost (EUR)",            "Value": round(total_setup, 2)},
        {"Metric": "Holding Cost (EUR)",          "Value": round(total_holding, 2)},
        {"Metric": "Overtime Cost X (EUR)",       "Value": round(total_ot_x, 2)},
        {"Metric": "Overtime Cost Y (EUR)",       "Value": round(total_ot_y, 2)},
        {"Metric": "Modernization Cost X (EUR)",  "Value": round(mod_cost_x, 2)},
        {"Metric": "Modernization Cost Y (EUR)",  "Value": round(mod_cost_y, 2)},
        {"Metric": "Backorder Cost (EUR)",        "Value": round(total_bo, 2)},
        {"Metric": "Total Cost (EUR)",            "Value": round(total_cost, 2)},
        {"Metric": "Added capacity WS-X (units)", "Value": round(dx_fix, 2)},
        {"Metric": "Added capacity WS-Y (%)",     "Value": round(dy_fix, 4)},
        {"Metric": "New capacity WS-X (units)",   "Value": round(new_cap_x, 2)},
        {"Metric": "New capacity WS-Y (min)",     "Value": round(new_cap_y, 2)},
        {"Metric": "Service Level",               "Value": round(service_level, 4)},
        {"Metric": "Fill Rate",                   "Value": round(fill_rate, 4)},
        {"Metric": "Rolling Horizon Frequency",   "Value": rh_freq},
        {"Metric": "Safety Stock E2801 (units)",  "Value": ss},
    ]
    df_summary = pd.DataFrame(summary_rows).set_index("Metric")

    # Production plan
    prod_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row["W" + str(t)] = p_executed.get((i, t), 0)
        prod_rows.append(row)
    df_prod = pd.DataFrame(prod_rows).set_index("Part")

    # Inventory plan
    inv_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row["W" + str(t)] = inv_real.get((i, t), 0)
        inv_rows.append(row)
    df_inv = pd.DataFrame(inv_rows).set_index("Part")

    # Setup decisions
    setup_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row["W" + str(t)] = y_executed.get((i, t), 0)
        setup_rows.append(row)
    df_setup = pd.DataFrame(setup_rows).set_index("Part")

    # Backorders
    bo_row = {"Part": "Backorder E2801"}
    for t in periods:
        bo_row["W" + str(t)] = int(round(bo_real.get(t, 0)))
    df_bo = pd.DataFrame([bo_row]).set_index("Part")

    # Overtime
    ot_rows = []
    for t in periods:
        ox_t = ox_executed.get(t, 0)
        oy_t = oy_executed.get(t, 0)
        ot_rows.append({
            "Period":          t,
            "OT Units X":      round(ox_t, 1),
            "OT Minutes Y":    round(oy_t, 1),
            "OT Hours Y":      round(oy_t / 60.0, 2),
            "OT Cost X (EUR)": round(OT_COST_X * ox_t, 2),
            "OT Cost Y (EUR)": round(OT_COST_Y * (oy_t / 60.0), 2),
        })
    df_ot = pd.DataFrame(ot_rows).set_index("Period")

    # Demand rows
    df_dem_fcst = demand_row_df(D_fcst, periods, label="Forecast Demand E2801")
    df_dem_real = demand_row_df(D_real, periods, label="Realized Demand E2801")

    # Method summary
    method_rows = [
        {"Item": "Method",
         "Description": f"Rolling Horizon (re-plan every {rh_freq} period(s)) "
                        f"+ Safety Stock (SS={ss} units for E2801)"},
        {"Item": "Why Rolling Horizon",
         "Description": "Re-optimizing regularly absorbs forecast errors by "
                        "updating plans with actual inventory levels, preventing "
                        "the demand drift that caused backorders in week 26 & 30 in 5b."},
        {"Item": "Why Safety Stock",
         "Description": f"Within each window, future demand is still uncertain. "
                        f"SS={ss} units (≈95% CSL, based on sigma_error={24.9:.1f}) "
                        "buffers small within-window surprises without incurring backorders."},
        {"Item": "Modernization",
         "Description": f"Fixed from 5a: WS-X +{dx_fix:.0f} units, WS-Y +{dy_fix:.4f}%. "
                        f"One-time cost EUR {mod_cost_x+mod_cost_y:,.2f}."},
        {"Item": "Overtime",
         "Description": "Re-optimized every window, so overtime is only used "
                        "when truly needed given actual inventory state."},
        {"Item": "5b Total Cost (EUR)",   "Description": str(553686.60)},
        {"Item": "6  Total Cost (EUR)",   "Description": str(round(total_cost, 2))},
        {"Item": "Cost improvement (EUR)","Description": str(round(553686.60 - total_cost, 2))},
        {"Item": "5b Service Level",      "Description": "93.33%"},
        {"Item": "6  Service Level",      "Description": f"{service_level:.2%}"},
        {"Item": "5b Fill Rate",          "Description": "97.52%"},
        {"Item": "6  Fill Rate",          "Description": f"{fill_rate:.2%}"},
    ]
    df_method = pd.DataFrame(method_rows).set_index("Item")

    # ------------------------------------------------------------------
    # Step 5: Print + write
    # ------------------------------------------------------------------
    if print_summary:
        print("\n" + "=" * 70)
        print("  ASSIGNMENT 6 — Rolling Horizon + Safety Stock — RESULTS")
        print("=" * 70)
        print(df_cost.to_string())
        print("=" * 70)
        print(f"  Setup Cost        : EUR {total_setup:>12,.2f}")
        print(f"  Holding Cost      : EUR {total_holding:>12,.2f}")
        print(f"  Overtime Cost X   : EUR {total_ot_x:>12,.2f}")
        print(f"  Overtime Cost Y   : EUR {total_ot_y:>12,.2f}")
        print(f"  Modernization X   : EUR {mod_cost_x:>12,.2f}")
        print(f"  Modernization Y   : EUR {mod_cost_y:>12,.2f}")
        print(f"  Backorder Cost    : EUR {total_bo:>12,.2f}")
        print(f"  TOTAL COST        : EUR {total_cost:>12,.2f}")
        print(f"  Service Level     :       {service_level:.2%}")
        print(f"  Fill Rate         :       {fill_rate:.2%}")
        print()
        print(f"  Comparison vs 5b:")
        print(f"    5b Total Cost   : EUR   553,686.60")
        print(f"    6  Total Cost   : EUR {total_cost:>12,.2f}")
        print(f"    Improvement     : EUR {553686.60 - total_cost:>12,.2f}")

    write_excel(output_filename, {
        "Method Summary":   df_method,
        "Summary":          df_summary,
        "Cost per Part":    df_cost,
        "Overtime":         df_ot,
        "Production Plan":  pd.concat([df_prod, df_dem_fcst, df_dem_real]),
        "Inventory Plan":   df_inv,
        "Setup Decisions":  df_setup,
        "Backorders":       df_bo,
    })

    return {
        "total_cost":    total_cost,
        "service_level": service_level,
        "fill_rate":     fill_rate,
        "df_summary":    df_summary,
        "df_cost":       df_cost,
    }


# =============================================================================
# Demand row helper (not imported from utils to keep self-contained)
# =============================================================================

def demand_row_df(demand, periods, label):
    row = {"Part": label}
    for t in periods:
        row["W" + str(t)] = demand[t - 1]
    return pd.DataFrame([row]).set_index("Part")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    solve_6(
        rh_freq          = RH_FREQ,
        ss               = SS_E2801,
        output_filename  = "output_6.xlsx",
        print_summary    = True,
    )