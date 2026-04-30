"""
rolling_horizon.py
==================
Rolling-horizon production planning MIP for the APM project.

This file is compatible with the current project structure:
- data is loaded via utils.load_data()
- BOM is the nested BOM from utils.py
- production released in week t arrives in week t + LT[i]
- backorders are tracked only for E2801
- component shortfalls are not allowed
- dx is integer, as in Assignment 5
- dy is discretized using MOD_INCR_Y / MOD_COST_PCT_Y
- rolling horizon can be compared against the fixed 5a plan

Main rolling-horizon logic
--------------------------
At every rolling step:
1. solve a window MIP
2. commit only the frozen period(s)
3. update inventory with realized demand in realized mode
4. carry the pipeline and backlog forward

Backlog handling
----------------
For normal rolling windows, backlog is forced to be zero at the end of the
window. In the final tail of the horizon, this hard constraint is relaxed only
from DEFAULT_TAIL_RELAX_START onward, because lead times can make full recovery
physically impossible. Terminal backlog is still penalized strongly.

Typical use
-----------
Compare fixed 5a vs rolling horizon under realized demand:

    python rolling_horizon.py --compare-5a --mode realized --window 15 --frozen 1

Try a longer look-ahead:

    python rolling_horizon.py --compare-5a --mode realized --window 20 --frozen 1

Sensitivity analysis:

    python rolling_horizon.py --sensitivity --mode realized
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from utils import load_data, write_excel


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_PLANNING_BO_MULTIPLIER = 100.0
DEFAULT_TERMINAL_BO_MULTIPLIER = 10.0
DEFAULT_FORCE_CLEAR_BACKLOG = True
DEFAULT_TAIL_RELAX_START = 22

_REQUIRED_DATA_KEYS = [
    "parts", "periods", "D_fcst", "D_real",
    "LT", "I0", "BOM", "parents",
    "SC", "HC", "BO_COST",
    "CAP_X", "CAP_Y", "PROC_Y",
    "OT_COST_X", "OT_COST_Y", "OT_MAX_X", "OT_MAX_Y",
    "MOD_COST_X", "MOD_COST_PCT_Y",
    "MOD_MAX_X", "MOD_MAX_PCT_Y", "MOD_INCR_Y",
    "Q_min", "BIG_M",
]


# =============================================================================
# DATA CHECK
# =============================================================================

def _validate_data(data: dict) -> None:
    """Check that utils.load_data() returns all keys used in this file."""
    missing = [k for k in _REQUIRED_DATA_KEYS if k not in data]
    if missing:
        raise KeyError(
            "load_data() is missing required keys: "
            + str(missing)
            + "\nCheck utils.py and input_data.json."
        )


# =============================================================================
# REPORTING HELPERS
# =============================================================================

def _wide_df(hist: dict, parts: list, periods: list, integer: bool = True) -> pd.DataFrame:
    """Convert {part: {period: value}} to a wide DataFrame."""
    rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            val = hist[i][t]
            row["W" + str(t)] = int(round(val)) if integer else round(float(val), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Part")


def _backorder_df(backorders: dict, periods: list) -> pd.DataFrame:
    row = {"Part": "Backorder E2801"}
    for t in periods:
        row["W" + str(t)] = int(round(backorders[t]))
    return pd.DataFrame([row]).set_index("Part")


def _demand_df(data: dict) -> pd.DataFrame:
    periods = data["periods"]
    d_fcst = data["D_fcst"]
    d_real = data["D_real"]

    rows = [
        {
            "Part": "Forecast Demand E2801",
            **{"W" + str(t): d_fcst[t - 1] for t in periods},
        },
        {
            "Part": "Realized Demand E2801",
            **{"W" + str(t): d_real[t - 1] for t in periods},
        },
    ]
    return pd.DataFrame(rows).set_index("Part")


def _service_metrics(backorders: dict, demand: list, periods: list) -> tuple[float, float, float]:
    """
    Compute:
    - service level
    - fill rate
    - new backorder units

    backorders[t] is the end-of-period backlog stock.
    """
    periods_with_bo = sum(1 for t in periods if backorders[t] > 0.5)
    service_level = 1.0 - periods_with_bo / len(periods)

    total_demand = sum(demand)
    new_bo = 0.0

    for idx, t in enumerate(periods):
        prev_bo = backorders[periods[idx - 1]] if idx > 0 else 0.0
        new_bo += max(0.0, backorders[t] - prev_bo)

    fill_rate = 1.0 - new_bo / total_demand if total_demand > 0 else 1.0
    return service_level, fill_rate, new_bo


def _cost_summary_df(
    data: dict,
    setup_hist: dict,
    inv_hist: dict,
    extra: dict | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Create per-part and total cost summary."""
    parts = data["parts"]
    periods = data["periods"]
    sc = data["SC"]
    hc = data["HC"]

    rows: list[dict[str, Any]] = []
    total_setup = 0.0
    total_holding = 0.0

    for i in parts:
        setup_cost = sum(sc[i] * setup_hist[i][t] for t in periods)
        holding_cost = sum(hc[i] * inv_hist[i][t] for t in periods)
        num_setups = int(sum(setup_hist[i][t] for t in periods))

        total_setup += setup_cost
        total_holding += holding_cost

        rows.append({
            "Part": i,
            "Num Setups": num_setups,
            "Setup Cost (EUR)": round(setup_cost, 2),
            "Holding Cost (EUR)": round(holding_cost, 2),
            "Total Cost (EUR)": round(setup_cost + holding_cost, 2),
        })

    total_row: dict[str, Any] = {
        "Part": "TOTAL",
        "Num Setups": sum(r["Num Setups"] for r in rows),
        "Setup Cost (EUR)": round(total_setup, 2),
        "Holding Cost (EUR)": round(total_holding, 2),
        "Total Cost (EUR)": round(total_setup + total_holding, 2),
    }

    if extra:
        for col, val in extra.items():
            total_row[col] = round(val, 2)

    rows.append(total_row)
    return pd.DataFrame(rows).set_index("Part"), total_setup, total_holding


def _utilisation_df(
    data: dict,
    prod_hist: dict,
    ox_hist: dict,
    oy_hist: dict,
    dx: float,
    dy: float,
) -> pd.DataFrame:
    periods = data["periods"]
    cap_x = data["CAP_X"]
    cap_y = data["CAP_Y"]
    proc_y = data["PROC_Y"]

    rows = []

    for t in periods:
        x_used = prod_hist["E2801"][t]
        y_used = (
            proc_y["B1401"] * prod_hist["B1401"][t]
            + proc_y["B2302"] * prod_hist["B2302"][t]
        )

        x_cap = cap_x + dx + ox_hist.get(t, 0.0)
        y_cap = cap_y * (1.0 + dy / 100.0) + oy_hist.get(t, 0.0)

        rows.append({
            "Period": t,
            "WS-X Used (units)": round(x_used, 1),
            "WS-X Cap (units)": round(x_cap, 1),
            "WS-X Util (%)": round(x_used / x_cap * 100, 1) if x_cap > 0 else 0.0,
            "WS-Y Used (min)": round(y_used, 1),
            "WS-Y Cap (min)": round(y_cap, 1),
            "WS-Y Util (%)": round(y_used / y_cap * 100, 1) if y_cap > 0 else 0.0,
            "OT X (units)": round(ox_hist.get(t, 0.0), 1),
            "OT Y (min)": round(oy_hist.get(t, 0.0), 1),
        })

    return pd.DataFrame(rows).set_index("Period")


# =============================================================================
# FINALIZE RESULTS
# =============================================================================

def _finalise(
    data: dict,
    prod_hist: dict,
    setup_hist: dict,
    inv_hist: dict,
    backorders: dict,
    ox_hist: dict,
    oy_hist: dict,
    dx: float,
    dy: float,
    label: str,
    output_file: str | None,
    metric_demand: list,
    window: int | None = None,
    frozen: int | None = None,
) -> dict:
    periods = data["periods"]

    bo_cost = data["BO_COST"]
    ot_cost_x = data["OT_COST_X"]
    ot_cost_y = data["OT_COST_Y"]
    mod_cost_x_rate = data["MOD_COST_X"]
    mod_cost_y_rate = data["MOD_COST_PCT_Y"]

    total_ot_x = sum(ot_cost_x * ox_hist.get(t, 0.0) for t in periods)
    total_ot_y = sum(ot_cost_y * (oy_hist.get(t, 0.0) / 60.0) for t in periods)
    mod_cost_x = mod_cost_x_rate * dx
    mod_cost_y = mod_cost_y_rate * dy
    total_bo = sum(bo_cost * backorders[t] for t in periods)

    extra = {
        "Overtime Cost X (EUR)": total_ot_x,
        "Overtime Cost Y (EUR)": total_ot_y,
        "Modernization Cost X (EUR)": mod_cost_x,
        "Modernization Cost Y (EUR)": mod_cost_y,
        "Backorder Cost (EUR)": total_bo,
    }

    df_cost, total_setup, total_holding = _cost_summary_df(
        data, setup_hist, inv_hist, extra=extra
    )

    total_cost = (
        total_setup
        + total_holding
        + total_ot_x
        + total_ot_y
        + mod_cost_x
        + mod_cost_y
        + total_bo
    )

    df_cost.loc["TOTAL", "Grand Total (EUR)"] = round(total_cost, 2)

    service_level, fill_rate, new_bo = _service_metrics(
        backorders, metric_demand, periods
    )

    summary_rows = [
        ("Setup Cost (EUR)", total_setup),
        ("Holding Cost (EUR)", total_holding),
        ("Overtime Cost X (EUR)", total_ot_x),
        ("Overtime Cost Y (EUR)", total_ot_y),
        ("Modernization Cost X (EUR)", mod_cost_x),
        ("Modernization Cost Y (EUR)", mod_cost_y),
        ("Backorder Cost (EUR)", total_bo),
        ("Total Cost (EUR)", total_cost),
        ("Service Level", service_level),
        ("Fill Rate", fill_rate),
        ("New Backorders (units)", new_bo),
        ("dx (units)", dx),
        ("dy (%)", dy),
    ]

    if window is not None:
        summary_rows.append(("Window size", window))
    if frozen is not None:
        summary_rows.append(("Frozen periods", frozen))

    df_summary = pd.DataFrame(
        [{"Metric": m, "Value": round(v, 4)} for m, v in summary_rows]
    ).set_index("Metric")

    df_prod = _wide_df(prod_hist, data["parts"], periods)
    df_inv = _wide_df(inv_hist, data["parts"], periods)
    df_setup = _wide_df(setup_hist, data["parts"], periods)
    df_util = _utilisation_df(data, prod_hist, ox_hist, oy_hist, dx, dy)
    df_bo = _backorder_df(backorders, periods)
    df_dem = _demand_df(data)

    if output_file:
        write_excel(output_file, {
            "Summary": df_summary,
            "Cost per Part": df_cost,
            "Utilisation": df_util,
            "Production Plan": pd.concat([df_prod, df_dem]),
            "Inventory Plan": df_inv,
            "Setup Decisions": df_setup,
            "Backorders": df_bo,
        })
        print("  -> Output written to: " + output_file)

    return {
        "label": label,
        "total_cost": total_cost,
        "total_setup": total_setup,
        "total_holding": total_holding,
        "total_ot_x": total_ot_x,
        "total_ot_y": total_ot_y,
        "mod_cost_x": mod_cost_x,
        "mod_cost_y": mod_cost_y,
        "total_bo": total_bo,
        "service_level": service_level,
        "fill_rate": fill_rate,
        "new_bo": new_bo,
        "dx": dx,
        "dy": dy,
        "df_summary": df_summary,
        "df_cost": df_cost,
        "df_util": df_util,
        "df_prod": df_prod,
        "df_inv": df_inv,
        "df_setup": df_setup,
        "df_bo": df_bo,
        "prod_hist": prod_hist,
        "setup_hist": setup_hist,
        "inv_hist": inv_hist,
        "backorders": backorders,
        "ox_hist": ox_hist,
        "oy_hist": oy_hist,
    }


# =============================================================================
# FIXED PLAN SIMULATION
# =============================================================================

def simulate_fixed_plan(
    data: dict,
    p_plan: dict,
    y_plan: dict,
    ox_plan: dict,
    oy_plan: dict,
    dx: float,
    dy: float,
    demand: list,
    label: str = "Fixed plan",
    output_file: str | None = None,
) -> dict:
    _validate_data(data)

    parts = data["parts"]
    periods = data["periods"]
    lt = data["LT"]
    i0 = data["I0"]
    bom = data["BOM"]
    parents = data["parents"]
    period_set = set(periods)

    on_hand = {i: float(i0[i]) for i in parts}
    pipeline: dict[str, dict[int, float]] = {i: {} for i in parts}
    backlog = 0.0

    prod_hist = {
        i: {t: float(p_plan.get((i, t), 0.0)) for t in periods}
        for i in parts
    }

    setup_hist = {
        i: {t: int(round(y_plan.get((i, t), 0))) for t in periods}
        for i in parts
    }

    inv_hist = {i: {t: 0.0 for t in periods} for i in parts}
    backorders = {t: 0.0 for t in periods}

    for t in periods:
        arrivals = {i: pipeline[i].pop(t, 0.0) for i in parts}

        for i in parts:
            arr_t = t + lt[i]
            if arr_t in period_set:
                pipeline[i][arr_t] = pipeline[i].get(arr_t, 0.0) + prod_hist[i][t]

        next_on_hand = {}
        next_backlog = backlog

        for i in parts:
            if i == "E2801":
                net = on_hand[i] + arrivals[i] - backlog - demand[t - 1]

                if net >= 0.0:
                    next_on_hand[i] = net
                    next_backlog = 0.0
                else:
                    next_on_hand[i] = 0.0
                    next_backlog = -net
            else:
                ind_demand = (
                    sum(bom[j][i] * prod_hist[j][t] for j in parents[i])
                    if i in parents else 0.0
                )
                net = on_hand[i] + arrivals[i] - ind_demand

                if net < -1e-6:
                    raise RuntimeError(
                        f"Negative component inventory in fixed simulation: "
                        f"part={i}, period={t}, value={net:.4f}"
                    )

                next_on_hand[i] = max(0.0, net)

        on_hand = next_on_hand
        backlog = next_backlog

        for i in parts:
            inv_hist[i][t] = on_hand[i]
        backorders[t] = backlog

    return _finalise(
        data=data,
        prod_hist=prod_hist,
        setup_hist=setup_hist,
        inv_hist=inv_hist,
        backorders=backorders,
        ox_hist=dict(ox_plan),
        oy_hist=dict(oy_plan),
        dx=dx,
        dy=dy,
        label=label,
        output_file=output_file,
        metric_demand=demand,
    )


# =============================================================================
# MODERNIZATION DISCRETIZATION
# =============================================================================

def _dy_params(data: dict) -> tuple[float, int]:
    """
    Return:
    - percentage-point increase per modernization increment
    - maximum number of increments

    In utils.py:
    MOD_INCR_Y is the EUR amount per increment.
    MOD_COST_PCT_Y is the EUR cost per 1% capacity increase.

    Example:
    15 / 1500 = 0.01%
    """
    pct_per_increment = float(data["MOD_INCR_Y"]) / float(data["MOD_COST_PCT_Y"])
    max_pct = float(data["MOD_MAX_PCT_Y"])
    max_increments = int(round(max_pct / pct_per_increment))
    return pct_per_increment, max_increments


# =============================================================================
# WINDOW MIP
# =============================================================================

def _solve_window(
    data: dict,
    window_periods: list[int],
    opening_inv: dict,
    opening_backlog: float,
    pipeline_snapshot: dict,
    dx: float,
    dy: float,
    demand_plan: dict,
    dx_free: bool = False,
    time_limit: int = 60,
    mip_gap: float = 1e-4,
    allow_backorders: bool = True,
    planning_bo_multiplier: float = DEFAULT_PLANNING_BO_MULTIPLIER,
    terminal_bo_multiplier: float = DEFAULT_TERMINAL_BO_MULTIPLIER,
    force_clear_backlog: bool = DEFAULT_FORCE_CLEAR_BACKLOG,
    tail_relax_start: int = DEFAULT_TAIL_RELAX_START,
) -> dict:
    parts = data["parts"]
    lt = data["LT"]
    q_min = data["Q_min"]
    sc = data["SC"]
    hc = data["HC"]
    bom = data["BOM"]
    parents = data["parents"]
    big_m = data["BIG_M"]

    cap_x = data["CAP_X"]
    cap_y = data["CAP_Y"]
    proc_y = data["PROC_Y"]

    ot_cost_x = data["OT_COST_X"]
    ot_cost_y = data["OT_COST_Y"]
    ot_max_x = data["OT_MAX_X"]
    ot_max_y = data["OT_MAX_Y"]

    real_bo_cost = data["BO_COST"]
    planning_bo_cost = real_bo_cost * planning_bo_multiplier

    mod_cost_x_rate = data["MOD_COST_X"]
    mod_cost_y_rate = data["MOD_COST_PCT_Y"]
    mod_max_x = data["MOD_MAX_X"]

    dy_incr, dy_max_inc = _dy_params(data)

    first_t = window_periods[0]
    last_t = window_periods[-1]
    final_period = data["periods"][-1]
    win_set = set(window_periods)

    m = gp.Model("rolling_window")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)

    p = m.addVars(parts, window_periods, name="p", lb=0.0, vtype=GRB.INTEGER)
    q = m.addVars(parts, window_periods, name="q", lb=0.0, vtype=GRB.INTEGER)
    y = m.addVars(parts, window_periods, name="y", vtype=GRB.BINARY)
    b = m.addVars(window_periods, name="b", lb=0.0, vtype=GRB.INTEGER)

    ox = m.addVars(
        window_periods,
        name="ox",
        lb=0.0,
        ub=ot_max_x,
        vtype=GRB.INTEGER,
    )

    oy = m.addVars(
        window_periods,
        name="oy",
        lb=0.0,
        ub=ot_max_y,
    )

    # dx and dy are optimized only in the first window unless fixed from 5a.
    if dx_free:
        dx_var = m.addVar(lb=0.0, ub=mod_max_x, vtype=GRB.INTEGER, name="dx")

        dy_int = m.addVar(lb=0, ub=dy_max_inc, vtype=GRB.INTEGER, name="dy_int")
        dy_var = m.addVar(lb=0.0, ub=dy_max_inc * dy_incr, name="dy")

        m.addConstr(dy_var == dy_incr * dy_int, name="dy_discretization")

        modernization_cost = mod_cost_x_rate * dx_var + mod_cost_y_rate * dy_var
    else:
        dx_var = dx
        dy_var = dy
        modernization_cost = 0.0

    # Backlog policy:
    # - forecast mode: no backlog allowed
    # - realized mode: backlog allowed, but heavily penalized
    # - normal rolling windows: force backlog to zero at the end of the window
    # - final tail windows: relax the hard clear only from tail_relax_start onward
    #
    # Why this matters:
    # With window=15, every window from W16 onward ends at W30.
    # If we relax every W*-W30 window, the model starts deferring too much demand.
    # Therefore, W16-W30 through W21-W30 still require b[30] = 0.
    # From W22-W30 onward, the hard clear is relaxed because lead times may make
    # b[30] = 0 infeasible.
    if not allow_backorders:
        for t in window_periods:
            m.addConstr(b[t] == 0, name=f"no_backorder_{t}")
    elif force_clear_backlog and first_t < tail_relax_start:
        m.addConstr(b[last_t] == 0, name="clear_backlog_end_window")

    terminal_backlog_penalty = 0.0
    if allow_backorders and last_t == final_period:
        terminal_backlog_penalty = planning_bo_cost * terminal_bo_multiplier * b[last_t]

    objective = (
        gp.quicksum(sc[i] * y[i, t] + hc[i] * q[i, t] for i in parts for t in window_periods)
        + gp.quicksum(ot_cost_x * ox[t] for t in window_periods)
        + gp.quicksum(ot_cost_y * (oy[t] / 60.0) for t in window_periods)
        + gp.quicksum(planning_bo_cost * b[t] for t in window_periods)
        + terminal_backlog_penalty
        + modernization_cost
    )

    m.setObjective(objective, GRB.MINIMIZE)

    for i in parts:
        for t in window_periods:
            q_prev = opening_inv[i] if t == first_t else q[i, t - 1]

            arriving_committed = pipeline_snapshot[i].get(t, 0.0)

            t_order = t - lt[i]
            arriving_new = p[i, t_order] if t_order in win_set else 0.0

            arriving = arriving_committed + arriving_new

            ext_demand = demand_plan.get(t, 0.0) if i == "E2801" else 0.0

            ind_demand = (
                gp.quicksum(bom[j][i] * p[j, t] for j in parents[i])
                if i in parents else 0.0
            )

            if i == "E2801":
                b_prev = opening_backlog if t == first_t else b[t - 1]

                m.addConstr(
                    q_prev + arriving + b[t] == ext_demand + b_prev + q[i, t],
                    name=f"inv_{i}_{t}",
                )
            else:
                m.addConstr(
                    q_prev + arriving == ind_demand + q[i, t],
                    name=f"inv_{i}_{t}",
                )

            m.addConstr(p[i, t] >= q_min[i] * y[i, t], name=f"minlot_{i}_{t}")
            m.addConstr(p[i, t] <= big_m * y[i, t], name=f"force_{i}_{t}")

    for t in window_periods:
        m.addConstr(
            p["E2801", t] <= cap_x + dx_var + ox[t],
            name=f"cap_X_{t}",
        )

        m.addConstr(
            proc_y["B1401"] * p["B1401", t]
            + proc_y["B2302"] * p["B2302", t]
            <= cap_y * (1.0 + dy_var / 100.0) + oy[t],
            name=f"cap_Y_{t}",
        )

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        try:
            m.computeIIS()
            m.write("infeasible_window.ilp")
        except Exception:
            pass

        raise RuntimeError(
            f"Window W{first_t}-W{last_t} infeasible or unsolved. "
            f"Gurobi status={m.Status}. IIS written if available."
        )

    result = {
        "p": {(i, t): int(round(p[i, t].X)) for i in parts for t in window_periods},
        "y": {(i, t): int(round(y[i, t].X)) for i in parts for t in window_periods},
        "q": {(i, t): float(q[i, t].X) for i in parts for t in window_periods},
        "b": {t: float(b[t].X) for t in window_periods},
        "ox": {t: float(ox[t].X) for t in window_periods},
        "oy": {t: float(oy[t].X) for t in window_periods},
        "obj": float(m.ObjVal),
    }

    if dx_free:
        result["dx"] = float(dx_var.X)
        result["dy"] = float(dy_var.X)
    else:
        result["dx"] = dx
        result["dy"] = dy

    return result


# =============================================================================
# ROLLING HORIZON SOLVER
# =============================================================================

def solve_rolling(
    mode: str = "realized",
    window: int = 15,
    frozen: int = 1,
    dx_fixed: float | None = None,
    dy_fixed: float | None = None,
    time_limit: int = 60,
    mip_gap: float = 1e-4,
    current_demand_known: bool = True,
    planning_bo_multiplier: float = DEFAULT_PLANNING_BO_MULTIPLIER,
    terminal_bo_multiplier: float = DEFAULT_TERMINAL_BO_MULTIPLIER,
    force_clear_backlog: bool = DEFAULT_FORCE_CLEAR_BACKLOG,
    tail_relax_start: int = DEFAULT_TAIL_RELAX_START,
    output_file: str | None = None,
    print_progress: bool = True,
) -> dict:
    if mode not in {"forecast", "realized"}:
        raise ValueError("mode must be 'forecast' or 'realized'")

    if frozen < 1:
        raise ValueError("frozen must be >= 1")

    if window < frozen:
        raise ValueError("window must be >= frozen")

    if (dx_fixed is None) != (dy_fixed is None):
        raise ValueError("dx_fixed and dy_fixed must both be provided or both be None.")

    data = load_data()
    _validate_data(data)

    parts = data["parts"]
    periods = data["periods"]

    d_fcst = data["D_fcst"]
    d_real = data["D_real"]

    lt = data["LT"]
    i0 = data["I0"]
    bom = data["BOM"]
    parents = data["parents"]

    n = len(periods)
    period_set = set(periods)

    on_hand = {i: float(i0[i]) for i in parts}
    pipeline: dict[str, dict[int, float]] = {i: {} for i in parts}
    backlog = 0.0

    prod_hist = {i: {t: 0.0 for t in periods} for i in parts}
    setup_hist = {i: {t: 0 for t in periods} for i in parts}
    inv_hist = {i: {t: 0.0 for t in periods} for i in parts}
    backorders = {t: 0.0 for t in periods}
    ox_hist = {t: 0.0 for t in periods}
    oy_hist = {t: 0.0 for t in periods}

    dx_locked = dx_fixed
    dy_locked = dy_fixed

    idx = 0

    while idx < n:
        start_t = periods[idx]
        end_idx = min(idx + window, n)

        window_periods = periods[idx:end_idx]
        commit_count = min(frozen, n - idx)
        commit_periods = periods[idx:idx + commit_count]

        demand_plan = {}

        for t in window_periods:
            if mode == "realized" and current_demand_known and t == start_t:
                demand_plan[t] = float(d_real[t - 1])
            else:
                demand_plan[t] = float(d_fcst[t - 1])

        first_window = dx_locked is None

        if print_progress:
            dx_text = "free" if first_window else f"{dx_locked:.2f}"
            dy_text = "free" if first_window else f"{dy_locked:.4f}"

            print(
                f"[RH] Window W{window_periods[0]}-W{window_periods[-1]} | "
                f"commit {commit_periods} | dx={dx_text}, dy={dy_text}%"
            )

        pipeline_snapshot = {i: dict(pipeline[i]) for i in parts}

        sol = _solve_window(
            data=data,
            window_periods=window_periods,
            opening_inv=on_hand,
            opening_backlog=backlog,
            pipeline_snapshot=pipeline_snapshot,
            dx=dx_locked if dx_locked is not None else 0.0,
            dy=dy_locked if dy_locked is not None else 0.0,
            demand_plan=demand_plan,
            dx_free=first_window,
            time_limit=time_limit,
            mip_gap=mip_gap,
            allow_backorders=(mode == "realized"),
            planning_bo_multiplier=planning_bo_multiplier,
            terminal_bo_multiplier=terminal_bo_multiplier,
            force_clear_backlog=force_clear_backlog,
            tail_relax_start=tail_relax_start,
        )

        if first_window:
            dx_locked = sol["dx"]
            dy_locked = sol["dy"]

            if print_progress:
                print(
                    f"  Modernization locked: dx={dx_locked:.2f}, "
                    f"dy={dy_locked:.4f}%"
                )

        for t in commit_periods:
            p_exec = {i: sol["p"][(i, t)] for i in parts}
            y_exec = {i: sol["y"][(i, t)] for i in parts}

            for i in parts:
                prod_hist[i][t] = float(p_exec[i])
                setup_hist[i][t] = y_exec[i]

            ox_hist[t] = sol["ox"][t]
            oy_hist[t] = sol["oy"][t]

            arrivals = {i: pipeline[i].pop(t, 0.0) for i in parts}

            for i in parts:
                arr_t = t + lt[i]
                if arr_t in period_set:
                    pipeline[i][arr_t] = pipeline[i].get(arr_t, 0.0) + p_exec[i]

            actual_demand = d_real[t - 1] if mode == "realized" else d_fcst[t - 1]

            next_on_hand = {}
            next_backlog = backlog

            for i in parts:
                if i == "E2801":
                    net = on_hand[i] + arrivals[i] - backlog - actual_demand

                    if net >= 0.0:
                        next_on_hand[i] = net
                        next_backlog = 0.0
                    else:
                        next_on_hand[i] = 0.0
                        next_backlog = -net

                else:
                    ind_demand = (
                        sum(bom[j][i] * p_exec[j] for j in parents[i])
                        if i in parents else 0.0
                    )

                    net = on_hand[i] + arrivals[i] - ind_demand

                    if net < -1e-6:
                        raise RuntimeError(
                            f"Negative component inventory during rolling horizon: "
                            f"part={i}, period={t}, value={net:.4f}"
                        )

                    next_on_hand[i] = max(0.0, net)

            on_hand = next_on_hand
            backlog = next_backlog

            for i in parts:
                inv_hist[i][t] = on_hand[i]

            backorders[t] = backlog

        idx += commit_count

    metric_demand = d_real if mode == "realized" else d_fcst
    label = f"Rolling horizon (mode={mode}, W={window}, F={frozen})"

    result = _finalise(
        data=data,
        prod_hist=prod_hist,
        setup_hist=setup_hist,
        inv_hist=inv_hist,
        backorders=backorders,
        ox_hist=ox_hist,
        oy_hist=oy_hist,
        dx=dx_locked,
        dy=dy_locked,
        label=label,
        output_file=output_file,
        metric_demand=metric_demand,
        window=window,
        frozen=frozen,
    )

    if print_progress:
        _print_summary(result, data)

    return result


# =============================================================================
# COMPARISON AND SENSITIVITY
# =============================================================================

def compare_fixed_vs_rolling(
    fixed_result: dict,
    rolling_result: dict,
    output_file: str = "output_rh_comparison.xlsx",
) -> pd.DataFrame:
    keys = [
        "total_setup",
        "total_holding",
        "total_ot_x",
        "total_ot_y",
        "mod_cost_x",
        "mod_cost_y",
        "total_bo",
        "total_cost",
        "service_level",
        "fill_rate",
        "new_bo",
        "dx",
        "dy",
    ]

    rows = []

    for result in (fixed_result, rolling_result):
        row = {"Plan": result["label"]}
        for key in keys:
            row[key] = result[key]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Plan")

    diff = df.loc[rolling_result["label"]] - df.loc[fixed_result["label"]]
    diff.name = "Improvement (RH - Fixed)"

    df = pd.concat([df, diff.to_frame().T])

    write_excel(output_file, {"Comparison": df})
    print("  -> Comparison written to: " + output_file)

    return df


def sensitivity_analysis(
    configs: list[tuple[int, int]] | None = None,
    mode: str = "realized",
    dx_fixed: float | None = None,
    dy_fixed: float | None = None,
    time_limit: int = 60,
    mip_gap: float = 1e-4,
    planning_bo_multiplier: float = DEFAULT_PLANNING_BO_MULTIPLIER,
    terminal_bo_multiplier: float = DEFAULT_TERMINAL_BO_MULTIPLIER,
    force_clear_backlog: bool = DEFAULT_FORCE_CLEAR_BACKLOG,
    tail_relax_start: int = DEFAULT_TAIL_RELAX_START,
    output_file: str = "output_rh_sensitivity.xlsx",
) -> pd.DataFrame:
    if configs is None:
        configs = [
            (5, 1),
            (5, 2),
            (10, 1),
            (10, 3),
            (10, 5),
            (15, 1),
            (15, 5),
            (20, 1),
            (30, 30),
        ]

    records = []

    for win, frz in configs:
        print("\n" + "=" * 70)
        print(f"Config: window={win}, frozen={frz}, mode={mode}")
        print("=" * 70)

        result = solve_rolling(
            mode=mode,
            window=win,
            frozen=frz,
            dx_fixed=dx_fixed,
            dy_fixed=dy_fixed,
            time_limit=time_limit,
            mip_gap=mip_gap,
            planning_bo_multiplier=planning_bo_multiplier,
            terminal_bo_multiplier=terminal_bo_multiplier,
            force_clear_backlog=force_clear_backlog,
            tail_relax_start=tail_relax_start,
            output_file=None,
            print_progress=True,
        )

        records.append({
            "window": win,
            "frozen": frz,
            "dx": round(result["dx"], 2),
            "dy (%)": round(result["dy"], 4),
            "setup": round(result["total_setup"], 2),
            "holding": round(result["total_holding"], 2),
            "OT X": round(result["total_ot_x"], 2),
            "OT Y": round(result["total_ot_y"], 2),
            "backorder": round(result["total_bo"], 2),
            "total_cost": round(result["total_cost"], 2),
            "service_level": round(result["service_level"], 4),
            "fill_rate": round(result["fill_rate"], 4),
            "new_bo": round(result["new_bo"], 2),
        })

    df = pd.DataFrame(records).set_index(["window", "frozen"])

    if (30, 30) in df.index:
        df["vs_oneshot"] = df["total_cost"] - df.loc[(30, 30), "total_cost"]

    write_excel(output_file, {"Sensitivity": df})
    print("  -> Sensitivity output written to: " + output_file)

    return df


def _print_summary(result: dict, data: dict) -> None:
    print("\n" + "-" * 72)
    print(result["label"])
    print("-" * 72)
    print(f"Setup cost          : EUR {result['total_setup']:,.2f}")
    print(f"Holding cost        : EUR {result['total_holding']:,.2f}")
    print(f"Overtime cost X     : EUR {result['total_ot_x']:,.2f}")
    print(f"Overtime cost Y     : EUR {result['total_ot_y']:,.2f}")
    print(f"Modernization X     : EUR {result['mod_cost_x']:,.2f}")
    print(f"Modernization Y     : EUR {result['mod_cost_y']:,.2f}")
    print(f"Backorder cost      : EUR {result['total_bo']:,.2f}")
    print(f"Total cost          : EUR {result['total_cost']:,.2f}")
    print(f"Service level       : {result['service_level']:.2%}")
    print(f"Fill rate           : {result['fill_rate']:.2%}")
    print(f"New BO units        : {result['new_bo']:.0f}")
    print(f"dx                  : {result['dx']:.2f}")
    print(f"dy                  : {result['dy']:.4f}%")
    print(f"New WS-X capacity   : {data['CAP_X'] + result['dx']:.1f}")
    print(f"New WS-Y capacity   : {data['CAP_Y'] * (1 + result['dy'] / 100):.1f}")
    print("-" * 72)


# =============================================================================
# LOAD 5A PLAN
# =============================================================================

def load_5a_plan(module_path: str | None = None) -> dict:
    path = Path(module_path) if module_path else Path(__file__).with_name("5aFUNCTION.py")

    spec = importlib.util.spec_from_file_location("assignment5a", path)

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load module from " + str(path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.solve_5a_plan(
        write_output=False,
        output_filename="output_5a.xlsx",
        print_summary=False,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-horizon production planner")

    parser.add_argument("--mode", default="realized", choices=["forecast", "realized"])
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--frozen", type=int, default=1)
    parser.add_argument("--timelimit", type=int, default=60)
    parser.add_argument("--mipgap", type=float, default=1e-4)
    parser.add_argument("--bo-multiplier", type=float, default=DEFAULT_PLANNING_BO_MULTIPLIER)
    parser.add_argument("--terminal-bo-multiplier", type=float, default=DEFAULT_TERMINAL_BO_MULTIPLIER)
    parser.add_argument("--tail-relax-start", type=int, default=DEFAULT_TAIL_RELAX_START)
    parser.add_argument("--no-clear-window-backlog", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--compare-5a", action="store_true")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    force_clear = not args.no_clear_window_backlog

    if args.sensitivity:
        output_file = args.output or f"output_rh_sensitivity_{args.mode}.xlsx"

        sensitivity_analysis(
            mode=args.mode,
            time_limit=args.timelimit,
            mip_gap=args.mipgap,
            planning_bo_multiplier=args.bo_multiplier,
            terminal_bo_multiplier=args.terminal_bo_multiplier,
            force_clear_backlog=force_clear,
            tail_relax_start=args.tail_relax_start,
            output_file=output_file,
        )

    elif args.compare_5a:
        data = load_data()
        _validate_data(data)

        plan_5a = load_5a_plan()

        dx_5a = plan_5a["dx_fix"]
        dy_5a = plan_5a["dy_fix"]

        if args.output:
            base = Path(args.output)
            fixed_output = str(base.with_name(base.stem + "_fixed_5a" + base.suffix))
            rolling_output = str(base.with_name(base.stem + "_rolling" + base.suffix))
        else:
            fixed_output = "output_rh_fixed_5a.xlsx"
            rolling_output = f"output_rh_{args.mode}_w{args.window}_f{args.frozen}.xlsx"

        fixed_result = simulate_fixed_plan(
            data=data,
            p_plan=plan_5a["p_fix"],
            y_plan=plan_5a["y_fix"],
            ox_plan=plan_5a["ox_fix"],
            oy_plan=plan_5a["oy_fix"],
            dx=dx_5a,
            dy=dy_5a,
            demand=data["D_real"],
            label="Fixed 5a baseline",
            output_file=fixed_output,
        )

        rolling_result = solve_rolling(
            mode=args.mode,
            window=args.window,
            frozen=args.frozen,
            dx_fixed=dx_5a,
            dy_fixed=dy_5a,
            time_limit=args.timelimit,
            mip_gap=args.mipgap,
            planning_bo_multiplier=args.bo_multiplier,
            terminal_bo_multiplier=args.terminal_bo_multiplier,
            force_clear_backlog=force_clear,
            tail_relax_start=args.tail_relax_start,
            output_file=rolling_output,
        )

        compare_fixed_vs_rolling(
            fixed_result,
            rolling_result,
            output_file="output_rh_comparison.xlsx",
        )

    else:
        output_file = args.output or f"output_rh_{args.mode}_w{args.window}_f{args.frozen}.xlsx"

        solve_rolling(
            mode=args.mode,
            window=args.window,
            frozen=args.frozen,
            time_limit=args.timelimit,
            mip_gap=args.mipgap,
            planning_bo_multiplier=args.bo_multiplier,
            terminal_bo_multiplier=args.terminal_bo_multiplier,
            force_clear_backlog=force_clear,
            tail_relax_start=args.tail_relax_start,
            output_file=output_file,
        )
