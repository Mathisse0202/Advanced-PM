""""
Assignment 1a - Multi-stage MIP Production Planning (Infinite Capacity)
========================================================================
Place this file in the same folder as input_data.json and run:
    python assignment1a.py
"""
import json
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from collections import defaultdict

# =============================================================================
# 1. LOAD INPUT DATA
# =============================================================================

with open("input_data.json", "r") as f:
    data = json.load(f)

T      = data["planning_horizon"]
parts  = data["parts"]
LT     = data["lead_times"]
Q_min  = data["min_lot_sizes"]
I0     = data["initial_inventory"]
SC     = data["setup_costs"]
HC     = data["holding_costs"]
BOM    = data["bom"]
D_fcst = data["demand_forecast"]

periods = list(range(1, T + 1))

# Build parent lookup: parents[child] = {parent: qty_per_parent}
parents = defaultdict(dict)
for parent, children in BOM.items():
    for child, qty in children.items():
        parents[child][parent] = qty

BIG_M = sum(D_fcst) * 100

# =============================================================================
# 2. BUILD GUROBI MODEL
# =============================================================================

m = gp.Model("Assignment1a")
m.setParam("OutputFlag", 1)
m.setParam("MIPGap", 1e-4)

# Decision variables
# p[i,t] : production/order quantity of part i in period t  (>= 0)
# q[i,t] : inventory of part i at end of period t           (>= 0)
# y[i,t] : binary setup variable for part i in period t     in {0,1}
p = m.addVars(parts, periods, name="p", lb=0.0)
q = m.addVars(parts, periods, name="q", lb=0.0)
y = m.addVars(parts, periods, name="y", vtype=GRB.BINARY)

# =============================================================================
# 3. OBJECTIVE FUNCTION
# =============================================================================
# Minimise total setup costs + total inventory holding costs

m.setObjective(
    gp.quicksum(SC[i] * y[i, t] + HC[i] * q[i, t]
                for i in parts for t in periods),
    GRB.MINIMIZE
)

# =============================================================================
# 4. CONSTRAINTS
# =============================================================================

for i in parts:
    for t in periods:

        # --- Inventory balance constraint (Graves P8, eq. 5) -----------------
        # Previous inventory + production arriving this period
        #   = external demand + induced demand (BOM) + ending inventory
        #
        # Production ordered in period (t - LT[i]) arrives in period t.
        q_prev = I0[i] if t == 1 else q[i, t - 1]
        t_order = t - LT[i]
        production_arriving = p[i, t_order] if t_order >= 1 else 0.0

        # External demand: only for the end product E2801
        ext_demand = D_fcst[t - 1] if i == "E2801" else 0

        # Induced demand: units of i required by production of parent parts
        induced_demand = (
            gp.quicksum(BOM[j][i] * p[j, t] for j in parents[i])
            if i in parents else 0
        )

        m.addConstr(
            q_prev + production_arriving == ext_demand + induced_demand + q[i, t],
            name=f"inv_balance_{i}_{t}"
        )

        # --- Minimum lot size + forcing constraints (Graves P8, eq. 7) -------
        # If y[i,t] = 1: p[i,t] >= Q_min[i]   (minimum lot size)
        # If y[i,t] = 0: p[i,t] = 0            (no production without setup)
        m.addConstr(p[i, t] >= Q_min[i] * y[i, t], name=f"min_lot_{i}_{t}")
        m.addConstr(p[i, t] <= BIG_M    * y[i, t], name=f"forcing_{i}_{t}")

# =============================================================================
# 5. SOLVE
# =============================================================================

m.optimize()

# =============================================================================
# 6. REPORT RESULTS
# =============================================================================

if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

    total_setup   = sum(SC[i] * y[i, t].X for i in parts for t in periods)
    total_holding = sum(HC[i] * q[i, t].X for i in parts for t in periods)
    total_cost    = total_setup + total_holding

    print("\n" + "=" * 60)
    print("  ASSIGNMENT 1a - OPTIMAL SOLUTION")
    print("=" * 60)
    print(f"\n  Total Setup Cost    : EUR {total_setup:>12,.2f}")
    print(f"  Total Holding Cost  : EUR {total_holding:>12,.2f}")
    print(f"  Total Cost          : EUR {total_cost:>12,.2f}")
    print(f"  MIP Gap             :  {m.MIPGap * 100:.4f}%")

    print(f"\n  {'Part':<10} {'# Setups':>10} {'Setup Cost':>14} {'Holding Cost':>14} {'Total':>14}")
    print("  " + "-" * 65)
    for i in parts:
        s  = sum(SC[i] * y[i, t].X for t in periods)
        h  = sum(HC[i] * q[i, t].X for t in periods)
        ns = int(sum(round(y[i, t].X) for t in periods))
        print(f"  {i:<10} {ns:>10} {s:>14,.2f} {h:>14,.2f} {s+h:>14,.2f}")
    print("  " + "-" * 65)
    print(f"  {'TOTAL':<10} {'':>10} {total_setup:>14,.2f} {total_holding:>14,.2f} {total_cost:>14,.2f}")

    # =========================================================================
    # 7. WRITE OUTPUT TO EXCEL
    # =========================================================================

    # Cost summary table
    cost_rows = []
    for i in parts:
        s  = sum(SC[i] * y[i, t].X for t in periods)
        h  = sum(HC[i] * q[i, t].X for t in periods)
        ns = int(sum(round(y[i, t].X) for t in periods))
        cost_rows.append({
            "Part": i,
            "# Setups": ns,
            "Setup Cost (EUR)": round(s, 2),
            "Holding Cost (EUR)": round(h, 2),
            "Total Cost (EUR)": round(s + h, 2)
        })
    cost_rows.append({
        "Part": "TOTAL",
        "# Setups": sum(r["# Setups"] for r in cost_rows),
        "Setup Cost (EUR)": round(total_setup, 2),
        "Holding Cost (EUR)": round(total_holding, 2),
        "Total Cost (EUR)": round(total_cost, 2)
    })
    df_cost = pd.DataFrame(cost_rows).set_index("Part")

    # Helper to build a wide dataframe (parts x periods)
    def make_df(var_dict):
        rows = []
        for i in parts:
            row = {"Part": i}
            for t in periods:
                row[f"W{t}"] = round(var_dict[i, t].X, 1)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Part")

    df_prod  = make_df(p)
    df_inv   = make_df(q)

    setup_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row[f"W{t}"] = int(round(y[i, t].X))
        setup_rows.append(row)
    df_setup = pd.DataFrame(setup_rows).set_index("Part")

    # Add demand row to production sheet
    demand_row = {"Part": "Demand E2801"}
    for t in periods:
        demand_row[f"W{t}"] = D_fcst[t - 1]
    df_demand = pd.DataFrame([demand_row]).set_index("Part")
    df_prod_with_demand = pd.concat([df_prod, df_demand])

    output_file = "output_assignment1a.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_cost.to_excel(writer,             sheet_name="Cost Summary")
        df_prod_with_demand.to_excel(writer, sheet_name="Production Plan")
        df_inv.to_excel(writer,              sheet_name="Inventory Plan")
        df_setup.to_excel(writer,            sheet_name="Setup Decisions")

    print(f"\n  Output written to: {output_file}")
    print("=" * 60 + "\n")

else:
    print(f"\nModel not solved. Gurobi status: {m.Status}")