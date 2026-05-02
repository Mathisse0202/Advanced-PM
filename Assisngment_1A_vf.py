"""
assignment_1a.py
----------------
Assignment 1a: Multi-stage MIP production planning under infinite capacity.

No capacity constraints on workstations.
Minimizes total setup and holding costs over a 30-week horizon.
Uses forecasted demand (D_fcst).
"""

import json
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


# Load input data

with open("input_data.json", "r") as f:
    data = json.load(f)

T = data["planning_horizon"]
parts = data["parts"]
LT = data["lead_times"]
Q_min = data["min_lot_sizes"]
I0 = data["initial_inventory"]
SC = data["setup_costs"]
HC = data["holding_costs"]
BOM = data["bom"]
D_fcst = data["demand_forecast"]

periods = list(range(1, T + 1))

# Reverse BOM: parents[child] = {parent: qty}
parents = defaultdict(dict)
for parent, children in BOM.items():
    for child, qty in children.items():
        parents[child][parent] = qty

BIG_M = sum(D_fcst) * 100


# Build model


m = gp.Model("Assignment1a")
m.setParam("OutputFlag", 1)
m.setParam("MIPGap", 1e-4)

# Decision variables
p = m.addVars(parts, periods, name="p", lb=0.0)
q = m.addVars(parts, periods, name="q", lb=0.0)
y = m.addVars(parts, periods, name="y", vtype=GRB.BINARY)


# Objective: minimize total setup + holding cost
m.setObjective(
    gp.quicksum(
        SC[i] * y[i, t] + HC[i] * q[i, t]
        for i in parts
        for t in periods
    ),
    GRB.MINIMIZE
)


# Constraints
for i in parts:
    for t in periods:
        q_prev = I0[i] if t == 1 else q[i, t - 1]

        t_order = t - LT[i]
        arriving = p[i, t_order] if t_order >= 1 else 0.0

        ext_demand = D_fcst[t - 1] if i == "E2801" else 0.0

        if i in parents:
            induced_demand = gp.quicksum(BOM[parent][i] * p[parent, t] for parent in parents[i])
        else:
            induced_demand = 0.0

        m.addConstr(
            q_prev + arriving == ext_demand + induced_demand + q[i, t],
            name=f"inv_balance_{i}_{t}"
        )

        # Inventory balance
        m.addConstr(
            p[i, t] >= Q_min[i] * y[i, t],
            name=f"min_lot_{i}_{t}"
        )

        # Minimum lot size
        m.addConstr(
            p[i, t] <= BIG_M * y[i, t],
            name=f"forcing_{i}_{t}"
        )


m.optimize()


if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:

    total_setup = sum(SC[i] * y[i, t].X for i in parts for t in periods)
    total_holding = sum(HC[i] * q[i, t].X for i in parts for t in periods)
    total_cost = total_setup + total_holding

    print("\n" + "=" * 70)
    print("ASSIGNMENT 1A - OPTIMAL SOLUTION")
    print("=" * 70)
    print(f"Total Setup Cost   : EUR {total_setup:,.2f}")
    print(f"Total Holding Cost : EUR {total_holding:,.2f}")
    print(f"Total Cost         : EUR {total_cost:,.2f}")
    print(f"MIP Gap            : {m.MIPGap * 100:.4f}%")
    print("=" * 70)

    # Per-part cost summary
    cost_rows = []
    for i in parts:
        setup_cost_i = sum(SC[i] * y[i, t].X for t in periods)
        holding_cost_i = sum(HC[i] * q[i, t].X for t in periods)
        num_setups_i = int(sum(round(y[i, t].X) for t in periods))

        cost_rows.append({
            "Part": i,
            "Num Setups": num_setups_i,
            "Setup Cost (EUR)": round(setup_cost_i, 2),
            "Holding Cost (EUR)": round(holding_cost_i, 2),
            "Total Cost (EUR)": round(setup_cost_i + holding_cost_i, 2),
        })

    cost_rows.append({
        "Part": "TOTAL",
        "Num Setups": sum(row["Num Setups"] for row in cost_rows),
        "Setup Cost (EUR)": round(total_setup, 2),
        "Holding Cost (EUR)": round(total_holding, 2),
        "Total Cost (EUR)": round(total_cost, 2),
    })

    df_cost = pd.DataFrame(cost_rows).set_index("Part")

    # Production plan
    prod_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row[f"W{t}"] = round(p[i, t].X, 1)
        prod_rows.append(row)
    df_prod = pd.DataFrame(prod_rows).set_index("Part")

    # Add forecast demand row
    demand_row = {"Part": "Demand E2801"}
    for t in periods:
        demand_row[f"W{t}"] = D_fcst[t - 1]
    df_demand = pd.DataFrame([demand_row]).set_index("Part")
    df_prod_with_demand = pd.concat([df_prod, df_demand])

    # Inventory plan
    inv_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row[f"W{t}"] = round(q[i, t].X, 1)
        inv_rows.append(row)
    df_inv = pd.DataFrame(inv_rows).set_index("Part")

    # Setup decisions
    setup_rows = []
    for i in parts:
        row = {"Part": i}
        for t in periods:
            row[f"W{t}"] = int(round(y[i, t].X))
        setup_rows.append(row)
    df_setup = pd.DataFrame(setup_rows).set_index("Part")

    # Write output file
    output_file = "output_assignment1a.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_cost.to_excel(writer, sheet_name="Cost Summary")
        df_prod_with_demand.to_excel(writer, sheet_name="Production Plan")
        df_inv.to_excel(writer, sheet_name="Inventory Plan")
        df_setup.to_excel(writer, sheet_name="Setup Decisions")

    print(f"\nOutput written to: {output_file}\n")

else:
    print(f"\nModel not solved. Gurobi status: {m.Status}")