"""
Assignment 4b
=============
Uses the FIXED production plan from 4a, evaluated against REALIZED demand.
Backorders at EUR 250 / unit / period where plan cannot meet demand.

Output: output_4b.xlsx
"""

import pandas as pd
import numpy as np
from utils import (
    load_data,
    build_cost_summary,
    compute_service_metrics,
    make_setup_df,
    demand_row_df,
    write_excel,
    print_cost_summary,
)

data           = load_data()
parts          = data["parts"]
periods        = data["periods"]
BO_COST        = data["BO_COST"]
MOD_COST_X     = data["MOD_COST_X"]
MOD_COST_PCT_Y = data["MOD_COST_PCT_Y"]
CAP_X          = data["CAP_X"]
CAP_Y          = data["CAP_Y"]
LT             = data["LT"]
I0             = data["I0"]
BOM            = data["BOM"]
HC             = data["HC"]
SC             = data["SC"]
Q_min          = data["Q_min"]
demand         = data["D_real"]

# --- Load fixed production plan from 4a ---
df_plan = pd.read_csv("plan_4a.csv", index_col=0)
# p_fixed[part][t] = production quantity
p_fixed = {
    i: {t: int(df_plan.loc[i, "W" + str(t)]) for t in periods}
    for i in parts
}

# --- Derive setup decisions from production plan ---
y_fixed = {
    i: {t: 1 if p_fixed[i][t] > 0 else 0 for t in periods}
    for i in parts
}

# --- Simulate inventory and backorders under realized demand ---
parents = data["parents"]
q_sim = {}   # inventory
b_sim = {}   # backorders (only for E2801)

for i in parts:
    for t in periods:
        q_prev = I0[i] if t == 1 else q_sim[i, t - 1]

        # Production arriving this period
        t_order = t - LT[i]
        arriving = p_fixed[i][t_order] if t_order >= 1 else 0

        if i == "E2801":
            b_prev = 0 if t == 1 else b_sim[t - 1]
            # Net inventory after fulfilling demand
            net = q_prev + arriving - b_prev - demand[t - 1]
            if net >= 0:
                q_sim[i, t] = net
                b_sim[t] = 0
            else:
                q_sim[i, t] = 0
                b_sim[t] = -net
        else:
            ind_demand = sum(
                BOM[j][i] * p_fixed[j][t]
                for j in parents[i]
                if j in BOM and i in BOM[j]
            )
            q_sim[i, t] = q_prev + arriving - ind_demand

# --- Compute costs ---
total_setup   = sum(SC[i] * y_fixed[i][t] for i in parts for t in periods)
total_holding = sum(HC[i] * q_sim[i, t] for i in parts for t in periods)
total_bo      = sum(BO_COST * b_sim[t] for t in periods)

# Load modernization values from 4a summary
df_summary_4a = pd.read_excel("output_4a.xlsx", sheet_name="Summary", index_col=0)
mod_cost_x = df_summary_4a.loc["Modernization Cost X (EUR)", "Value"]
mod_cost_y = df_summary_4a.loc["Modernization Cost Y (EUR)", "Value"]
dx_val     = df_summary_4a.loc["Added capacity WS-X (units)", "Value"]
dy_val     = df_summary_4a.loc["Added capacity WS-Y (%)", "Value"]
new_cap_x  = df_summary_4a.loc["New capacity WS-X (units)", "Value"]
new_cap_y  = df_summary_4a.loc["New capacity WS-Y (min/wk)", "Value"]

total_cost = total_setup + total_holding + mod_cost_x + mod_cost_y + total_bo

# --- Service metrics ---
periods_with_bo = sum(1 for t in periods if b_sim[t] > 0.5)
service_level   = 1.0 - periods_with_bo / len(periods)
total_demand    = sum(demand)
new_backorders  = sum(
    max(0, b_sim[t] - (b_sim[t - 1] if t > 1 else 0))
    for t in periods
)
fill_rate = 1.0 - new_backorders / total_demand

# --- Console output ---
print("\n" + "=" * 70)
print("  ASSIGNMENT 4B — Fixed Plan from 4a + Realized Demand + Backorders")
print("=" * 70)
print("  Setup Cost         : EUR " + "{:,.2f}".format(total_setup))
print("  Holding Cost       : EUR " + "{:,.2f}".format(total_holding))
print("  Backorder Cost     : EUR " + "{:,.2f}".format(total_bo))
print("  Modernization WS-X : EUR " + "{:,.2f}".format(mod_cost_x))
print("  Modernization WS-Y : EUR " + "{:,.2f}".format(mod_cost_y))
print("  Total Cost         : EUR " + "{:,.2f}".format(total_cost))
print("  New capacity WS-X  :  " + "{:.1f}".format(new_cap_x) + " units/week")
print("  New capacity WS-Y  :  " + "{:.1f}".format(new_cap_y) + " min/week")
print("  Service Level      :  " + "{:.2%}".format(service_level))
print("  Fill Rate          :  " + "{:.2%}".format(fill_rate))
print("=" * 70)

# --- Build output DataFrames ---
summary_rows = [
    {"Metric": "Setup Cost (EUR)",            "Value": round(total_setup, 2)},
    {"Metric": "Holding Cost (EUR)",          "Value": round(total_holding, 2)},
    {"Metric": "Backorder Cost (EUR)",        "Value": round(total_bo, 2)},
    {"Metric": "Modernization Cost X (EUR)",  "Value": round(mod_cost_x, 2)},
    {"Metric": "Modernization Cost Y (EUR)",  "Value": round(mod_cost_y, 2)},
    {"Metric": "Total Cost (EUR)",            "Value": round(total_cost, 2)},
    {"Metric": "Added capacity WS-X (units)", "Value": round(dx_val, 2)},
    {"Metric": "Added capacity WS-Y (%)",     "Value": round(dy_val, 4)},
    {"Metric": "New capacity WS-X (units)",   "Value": round(new_cap_x, 1)},
    {"Metric": "New capacity WS-Y (min/wk)",  "Value": round(new_cap_y, 1)},
    {"Metric": "Service Level",               "Value": round(service_level, 4)},
    {"Metric": "Fill Rate",                   "Value": round(fill_rate, 4)},
]
df_summary = pd.DataFrame(summary_rows).set_index("Metric")

# Production plan (same as 4a)
df_prod = df_plan.copy()

# Inventory plan
inv_rows = []
for i in parts:
    row = {"Part": i}
    for t in periods:
        row["W" + str(t)] = int(round(q_sim[i, t]))
    inv_rows.append(row)
df_inv = pd.DataFrame(inv_rows).set_index("Part")

# Setup decisions
setup_rows = []
for i in parts:
    row = {"Part": i}
    for t in periods:
        row["W" + str(t)] = y_fixed[i][t]
    setup_rows.append(row)
df_setup = pd.DataFrame(setup_rows).set_index("Part")

df_dem = demand_row_df(demand, periods, label="Realized Demand E2801")

write_excel("output_4b.xlsx", {
    "Summary":         df_summary,
    "Production Plan": pd.concat([df_prod, df_dem]),
    "Inventory Plan":  df_inv,
    "Setup Decisions": df_setup,
})
