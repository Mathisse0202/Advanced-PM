"""
utils.py
--------
Shared data loading and helper functions for APM Project 2026.
All assignments import from this file.
 
Usage:
    from utils import load_data, build_base_model, add_backorder_vars,
                      add_capacity_constraints, add_overtime_vars,
                      add_modernization_vars, compute_service_metrics,
                      make_plan_df, make_setup_df, build_cost_summary,
                      write_excel
"""
 
import json
import pandas as pd
from collections import defaultdict
 
 
# =============================================================================
# DATA LOADING
# =============================================================================
 
def load_data(path="input_data.json"):
    """Load all parameters from the JSON input file."""
    with open(path, "r") as f:
        raw = json.load(f)
 
    T       = raw["planning_horizon"]
    parts   = raw["parts"]
    LT      = raw["lead_times"]
    Q_min   = raw["min_lot_sizes"]
    I0      = raw["initial_inventory"]
    SC      = raw["setup_costs"]
    HC      = raw["holding_costs"]
    BOM     = raw["bom"]
    D_fcst  = raw["demand_forecast"]
    D_real  = raw["demand_realized"]
 
    # Capacity parameters
    CAP_X     = raw["workstation_X"]["capacity_per_week"]
    CAP_Y     = raw["workstation_Y"]["available_minutes_per_week"]
    PROC_Y    = raw["workstation_Y"]["processing_time_minutes"]  # {part: minutes}
 
    # Overtime parameters
    OT_COST_X = raw["overtime"]["cost_per_unit_X"]
    OT_MAX_X  = raw["overtime"]["max_units_X"]
    OT_COST_Y = raw["overtime"]["cost_per_hour_Y"]
    OT_MAX_Y  = raw["overtime"]["max_hours_Y"] * 60  # stored as minutes internally
 
    # Modernization parameters
    MOD_COST_X    = raw["modernization"]["cost_per_unit_X"]
    MOD_MAX_X     = raw["modernization"]["max_units_X"]
    MOD_COST_PCT_Y = raw["modernization"]["cost_per_pct_Y"]
    MOD_INCR_Y    = raw["modernization"]["increment_eur_Y"]   # EUR 15 per increment
    MOD_MAX_PCT_Y = raw["modernization"]["max_pct_Y"]
 
    # Backorder cost
    BO_COST = raw["backorder_cost_per_unit_per_period"]
 
    periods = list(range(1, T + 1))
 
    # Build reverse BOM: parents[child] = {parent: qty}
    parents = defaultdict(dict)
    for parent, children in BOM.items():
        for child, qty in children.items():
            parents[child][parent] = qty
 
    # Big-M for forcing constraints: safely larger than any feasible production
    BIG_M = sum(D_fcst) * 100
 
    return {
        "T": T,
        "parts": parts,
        "periods": periods,
        "LT": LT,
        "Q_min": Q_min,
        "I0": I0,
        "SC": SC,
        "HC": HC,
        "BOM": BOM,
        "parents": parents,
        "D_fcst": D_fcst,
        "D_real": D_real,
        "BIG_M": BIG_M,
        # Capacity
        "CAP_X": CAP_X,
        "CAP_Y": CAP_Y,
        "PROC_Y": PROC_Y,
        # Overtime
        "OT_COST_X": OT_COST_X,
        "OT_MAX_X": OT_MAX_X,
        "OT_COST_Y": OT_COST_Y,
        "OT_MAX_Y": OT_MAX_Y,
        # Modernization
        "MOD_COST_X": MOD_COST_X,
        "MOD_MAX_X": MOD_MAX_X,
        "MOD_COST_PCT_Y": MOD_COST_PCT_Y,
        "MOD_INCR_Y": MOD_INCR_Y,
        "MOD_MAX_PCT_Y": MOD_MAX_PCT_Y,
        # Backorder
        "BO_COST": BO_COST,
    }
 
 
# =============================================================================
# MODEL BUILDING BLOCKS
# =============================================================================
 
def build_base_model(data, demand, model_name, with_backorders=False):
    """
    Build the core MIP model (inventory balance + lot sizing).
 
    Parameters
    ----------
    data            : dict from load_data()
    demand          : list of length T — the demand to plan against
    model_name      : string label for Gurobi
    with_backorders : if True, add backorder variable b[t] for E2801
 
    Returns
    -------
    m      : Gurobi model (not yet optimized)
    p      : production variables  p[part, period]
    q      : inventory variables   q[part, period]
    y      : setup binary variables y[part, period]
    b      : backorder variables    b[period]  (None if with_backorders=False)
    """
    import gurobipy as gp
    from gurobipy import GRB
 
    parts   = data["parts"]
    periods = data["periods"]
    LT      = data["LT"]
    Q_min   = data["Q_min"]
    I0      = data["I0"]
    BOM     = data["BOM"]
    parents = data["parents"]
    BIG_M   = data["BIG_M"]
    SC      = data["SC"]
    HC      = data["HC"]
    BO_COST = data["BO_COST"]
 
    m = gp.Model(model_name)
    m.setParam("OutputFlag", 1)
    m.setParam("MIPGap", 1e-4)
 
    # --- Decision variables ---
    # Production and inventory quantities are integers (discrete units)
    p = m.addVars(parts, periods, name="p", lb=0.0, vtype=GRB.INTEGER)
    q = m.addVars(parts, periods, name="q", lb=0.0, vtype=GRB.INTEGER)
    y = m.addVars(parts, periods, name="y", vtype=GRB.BINARY)
    # Backorders are also integer (whole units)
    b = m.addVars(periods, name="b", lb=0.0, vtype=GRB.INTEGER) if with_backorders else None
 
    # --- Base objective: setup + holding costs ---
    base_obj = gp.quicksum(
        SC[i] * y[i, t] + HC[i] * q[i, t]
        for i in parts for t in periods
    )
    if with_backorders:
        base_obj = base_obj + gp.quicksum(BO_COST * b[t] for t in periods)
 
    # Objective will be set (or extended) by caller; set base here
    m.setObjective(base_obj, GRB.MINIMIZE)
 
    # --- Inventory balance constraints ---
    for i in parts:
        for t in periods:
            # Inventory at start of period
            q_prev = I0[i] if t == 1 else q[i, t - 1]
 
            # Production ordered in period (t - LT[i]) arrives now
            t_order = t - LT[i]
            arriving = p[i, t_order] if t_order >= 1 else 0.0
 
            # Demand composition
            ext_demand = demand[t - 1] if i == "E2801" else 0
            ind_demand = (
                gp.quicksum(BOM[j][i] * p[j, t] for j in parents[i])
                if i in parents else 0
            )
 
            if i == "E2801" and with_backorders:
                # Balance with backorders: supply = demand + new backlog + inventory
                b_prev = 0.0 if t == 1 else b[t - 1]
                m.addConstr(
                    q_prev + b_prev + arriving == ext_demand + b[t] + q[i, t],
                    name="inv_" + i + "_" + str(t)
                )
            else:
                m.addConstr(
                    q_prev + arriving == ext_demand + ind_demand + q[i, t],
                    name="inv_" + i + "_" + str(t)
                )
 
            # --- Lot sizing constraints ---
            # Minimum lot size: if setup, produce at least Q_min
            m.addConstr(
                p[i, t] >= Q_min[i] * y[i, t],
                name="minlot_" + i + "_" + str(t)
            )
            # Forcing: no production without a setup
            m.addConstr(
                p[i, t] <= BIG_M * y[i, t],
                name="force_" + i + "_" + str(t)
            )
 
    # All backorders must be cleared by end of horizon
    if with_backorders:
        m.addConstr(b[periods[-1]] == 0, name="no_final_backorder")
 
    return m, p, q, y, b
 
 
def add_capacity_constraints(m, p, data):
    """
    Add base capacity constraints for WS-X and WS-Y (no overtime, no modernization).
    Used in assignments 2a and 2b.
    """
    CAP_X  = data["CAP_X"]
    CAP_Y  = data["CAP_Y"]
    PROC_Y = data["PROC_Y"]
 
    for t in data["periods"]:
        m.addConstr(p["E2801", t] <= CAP_X, name="cap_X_" + str(t))
        m.addConstr(
            PROC_Y["B1401"] * p["B1401", t] + PROC_Y["B2302"] * p["B2302", t] <= CAP_Y,
            name="cap_Y_" + str(t)
        )
 
 
def add_overtime_vars(m, data):
    """
    Add overtime variables ox[t] and oy[t] to model m.
    Returns ox, oy.
    Used in assignments 3 and 5.
    """
    import gurobipy as gp
    periods   = data["periods"]
    OT_MAX_X  = data["OT_MAX_X"]
    OT_MAX_Y  = data["OT_MAX_Y"]   # in minutes
 
    # Overtime units on WS-X and WS-Y are both integer (whole units / whole minutes)
    ox = m.addVars(periods, name="ox", lb=0.0, ub=OT_MAX_X, vtype=GRB.INTEGER)
    oy = m.addVars(periods, name="oy", lb=0.0, ub=OT_MAX_Y, vtype=GRB.INTEGER)
    return ox, oy
 
 
def add_modernization_vars(m, data):
    """
    Add permanent modernization variables dx and dy to model m.
    dy is constrained to multiples of 0.01% (EUR 15 increments as per project spec).
    Returns dx, dy.
    Used in assignments 4 and 5.
    """
    import gurobipy as gp
    from gurobipy import GRB
 
    MOD_MAX_X     = data["MOD_MAX_X"]
    MOD_MAX_PCT_Y = data["MOD_MAX_PCT_Y"]
    MOD_INCR_Y    = data["MOD_INCR_Y"]       # EUR 15 per increment
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]  # EUR 1500 per 1%
 
    # Each increment of EUR 15 buys 15/1500 = 0.01% extra WS-Y capacity
    pct_per_increment = MOD_INCR_Y / MOD_COST_PCT_Y   # = 0.01
    max_increments    = int(MOD_MAX_PCT_Y / pct_per_increment)  # = 4000
 
    # dx: extra units of WS-X capacity — integer number of units
    # dy: returned as the integer increment variable (dy_int); callers multiply by
    #     pct_per_increment to get the actual % capacity increase.
    #     This keeps all decision variables integer and avoids fractional solutions.
    dx     = m.addVar(name="dx",     lb=0.0, ub=MOD_MAX_X,     vtype=GRB.INTEGER)
    dy_int = m.addVar(name="dy_int", lb=0.0, ub=max_increments, vtype=GRB.INTEGER)
 
    # Store the conversion factor on the model so capacity functions can use it
    m._pct_per_increment = pct_per_increment
 
    return dx, dy_int
 
 
def add_capacity_with_overtime(m, p, ox, oy, data):
    """
    Add capacity constraints extended with overtime.
    WS-X: p[E2801,t] <= CAP_X + ox[t]
    WS-Y: proc_B1401 * p[B1401,t] + proc_B2302 * p[B2302,t] <= CAP_Y + oy[t]
    Used in assignments 3 and 5.
    """
    CAP_X  = data["CAP_X"]
    CAP_Y  = data["CAP_Y"]
    PROC_Y = data["PROC_Y"]
 
    for t in data["periods"]:
        m.addConstr(p["E2801", t] <= CAP_X + ox[t], name="cap_X_" + str(t))
        m.addConstr(
            PROC_Y["B1401"] * p["B1401", t] + PROC_Y["B2302"] * p["B2302", t] <= CAP_Y + oy[t],
            name="cap_Y_" + str(t)
        )
 
 
def add_capacity_with_modernization(m, p, dx, dy, data):
    """
    Add capacity constraints extended with permanent modernization.
    WS-X: p[E2801,t] <= CAP_X + dx
    WS-Y: ... <= CAP_Y + (CAP_Y / 100) * pct_per_increment * dy
    dy here is dy_int (integer increments); pct_per_increment is stored on m.
    Used in assignment 4.
    """
    CAP_X  = data["CAP_X"]
    CAP_Y  = data["CAP_Y"]
    PROC_Y = data["PROC_Y"]
    pct_per_increment = m._pct_per_increment
 
    for t in data["periods"]:
        m.addConstr(p["E2801", t] <= CAP_X + dx, name="cap_X_" + str(t))
        m.addConstr(
            PROC_Y["B1401"] * p["B1401", t] + PROC_Y["B2302"] * p["B2302", t]
            <= CAP_Y + (CAP_Y / 100.0) * pct_per_increment * dy,
            name="cap_Y_" + str(t)
        )
 
 
def add_capacity_combined(m, p, ox, oy, dx, dy, data):
    """
    Add capacity constraints with BOTH overtime and modernization.
    WS-X: p[E2801,t] <= CAP_X + dx + ox[t]
    WS-Y: ... <= CAP_Y + (CAP_Y/100)*pct_per_increment*dy + oy[t]
    dy here is dy_int (integer increments); pct_per_increment is stored on m.
    Used in assignment 5.
    """
    CAP_X  = data["CAP_X"]
    CAP_Y  = data["CAP_Y"]
    PROC_Y = data["PROC_Y"]
    pct_per_increment = m._pct_per_increment
 
    for t in data["periods"]:
        m.addConstr(p["E2801", t] <= CAP_X + dx + ox[t], name="cap_X_" + str(t))
        m.addConstr(
            PROC_Y["B1401"] * p["B1401", t] + PROC_Y["B2302"] * p["B2302", t]
            <= CAP_Y + (CAP_Y / 100.0) * pct_per_increment * dy + oy[t],
            name="cap_Y_" + str(t)
        )
 
 
def set_overtime_objective(m, p, q, y, ox, oy, b, data, with_backorders):
    """
    Set full objective including overtime costs.
    Replaces the base objective set in build_base_model.
    """
    import gurobipy as gp
    from gurobipy import GRB
 
    parts     = data["parts"]
    periods   = data["periods"]
    SC        = data["SC"]
    HC        = data["HC"]
    BO_COST   = data["BO_COST"]
    OT_COST_X = data["OT_COST_X"]
    OT_COST_Y = data["OT_COST_Y"]
 
    obj = (
        gp.quicksum(SC[i] * y[i, t] + HC[i] * q[i, t] for i in parts for t in periods)
        + gp.quicksum(OT_COST_X * ox[t] for t in periods)
        + gp.quicksum(OT_COST_Y * (oy[t] / 60.0) for t in periods)
    )
    if with_backorders:
        obj = obj + gp.quicksum(BO_COST * b[t] for t in periods)
    m.setObjective(obj, GRB.MINIMIZE)
 
 
def set_modernization_objective(m, p, q, y, dx, dy, b, data, with_backorders):
    """
    Set full objective including modernization costs.
    dy is dy_int (integer increments); cost = MOD_COST_PCT_Y * pct_per_increment * dy.
    """
    import gurobipy as gp
    from gurobipy import GRB
 
    parts          = data["parts"]
    periods        = data["periods"]
    SC             = data["SC"]
    HC             = data["HC"]
    BO_COST        = data["BO_COST"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    pct_per_increment = m._pct_per_increment
 
    obj = (
        gp.quicksum(SC[i] * y[i, t] + HC[i] * q[i, t] for i in parts for t in periods)
        + MOD_COST_X * dx
        + MOD_COST_PCT_Y * pct_per_increment * dy
    )
    if with_backorders:
        obj = obj + gp.quicksum(BO_COST * b[t] for t in periods)
    m.setObjective(obj, GRB.MINIMIZE)
 
 
def set_combined_objective(m, p, q, y, ox, oy, dx, dy, b, data, with_backorders):
    """
    Set full objective: setup + holding + overtime + modernization + (optional) backorder.
    dy is dy_int (integer increments); cost = MOD_COST_PCT_Y * pct_per_increment * dy.
    oy is integer minutes; cost = OT_COST_Y * (oy / 60).
    """
    import gurobipy as gp
    from gurobipy import GRB
 
    parts          = data["parts"]
    periods        = data["periods"]
    SC             = data["SC"]
    HC             = data["HC"]
    BO_COST        = data["BO_COST"]
    OT_COST_X      = data["OT_COST_X"]
    OT_COST_Y      = data["OT_COST_Y"]
    MOD_COST_X     = data["MOD_COST_X"]
    MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
    pct_per_increment = m._pct_per_increment
 
    obj = (
        gp.quicksum(SC[i] * y[i, t] + HC[i] * q[i, t] for i in parts for t in periods)
        + gp.quicksum(OT_COST_X * ox[t] for t in periods)
        + gp.quicksum(OT_COST_Y * (oy[t] / 60.0) for t in periods)
        + MOD_COST_X * dx
        + MOD_COST_PCT_Y * pct_per_increment * dy
    )
    if with_backorders:
        obj = obj + gp.quicksum(BO_COST * b[t] for t in periods)
    m.setObjective(obj, GRB.MINIMIZE)
 
 
# =============================================================================
# RESULT EXTRACTION
# =============================================================================
 
def compute_service_metrics(b, demand, periods):
    """
    Compute cycle service level and fill rate from backorder solution values.
 
    Service level = fraction of periods with zero backorder.
    Fill rate     = fraction of demand units delivered on time.
    """
    periods_with_bo = sum(1 for t in periods if b[t].X > 0.5)
    service_level   = 1.0 - periods_with_bo / len(periods)
 
    total_demand = sum(demand)
    new_backorders = sum(
        max(0.0, b[t].X - (b[t - 1].X if t > 1 else 0.0))
        for t in periods
    )
    fill_rate = 1.0 - new_backorders / total_demand
 
    return service_level, fill_rate
 
 
def make_plan_df(var_dict, parts, periods):
    """Return a wide DataFrame: rows = parts, columns = W1..W30."""
    rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            val = var_dict[i, t].X
            # Show as integer when the value is whole (production/inventory are integer vars)
            row["W" + str(t)] = int(round(val))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Part")
 
 
def make_setup_df(y, parts, periods):
    """Return a wide DataFrame of binary setup decisions."""
    rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row["W" + str(t)] = int(round(y[i, t].X))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Part")
 
 
def build_cost_summary(p, q, y, data, extra=None):
    """
    Build per-part and total cost summary DataFrame.
 
    extra : dict of {column_name: value} to append to the TOTAL row
            (e.g. overtime costs, modernization costs, backorder costs)
    """
    parts   = data["parts"]
    periods = data["periods"]
    SC      = data["SC"]
    HC      = data["HC"]
 
    rows           = []
    total_setup    = 0.0
    total_holding  = 0.0
 
    for i in parts:
        s  = sum(SC[i] * y[i, t].X for t in periods)
        h  = sum(HC[i] * q[i, t].X for t in periods)
        ns = int(sum(round(y[i, t].X) for t in periods))
        total_setup   += s
        total_holding += h
        rows.append({
            "Part":               i,
            "Num Setups":         ns,
            "Setup Cost (EUR)":   round(s, 2),
            "Holding Cost (EUR)": round(h, 2),
            "Total Cost (EUR)":   round(s + h, 2),
        })
 
    total_row = {
        "Part":               "TOTAL",
        "Num Setups":         sum(r["Num Setups"] for r in rows),
        "Setup Cost (EUR)":   round(total_setup, 2),
        "Holding Cost (EUR)": round(total_holding, 2),
        "Total Cost (EUR)":   round(total_setup + total_holding, 2),
    }
    if extra:
        for col, val in extra.items():
            total_row[col] = round(val, 2)
 
    rows.append(total_row)
    return pd.DataFrame(rows).set_index("Part"), total_setup, total_holding
 
 
def demand_row_df(demand, periods, label="Demand E2801"):
    """Single-row DataFrame with demand per period, for appending to production plan."""
    row = {"Part": label}
    for t in periods:
        row["W" + str(t)] = demand[t - 1]
    return pd.DataFrame([row]).set_index("Part")
 
 
# =============================================================================
# OUTPUT
# =============================================================================
 
def write_excel(filename, sheets):
    """
    Write multiple DataFrames to an Excel file.
 
    sheets : dict {sheet_name: DataFrame}
    """
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name)
    print("  -> Output written to: " + filename)
 
 
def print_cost_summary(title, df_cost):
    """Print a formatted cost table to console."""
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)
    print(df_cost.to_string())
    print("=" * 70)