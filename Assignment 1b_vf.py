"""
Assignment 1b - Performance under realized demand

Reuses the fixed production plan from assignment_1a (output_assignment1a.xlsx).
Only E2801 inventory is recalculated; upstream holding costs remain unchanged.
Backorder cost: €250/unit/period.
Uses realized demand (D_real)
"""

import json
import pandas as pd
 
 
# Load input data
with open("input_data.json", "r") as f:
    data = json.load(f)
 
T       = data["planning_horizon"]
periods = list(range(1, T + 1))
I0      = data["initial_inventory"]
LT      = data["lead_times"]
HC      = data["holding_costs"]
BO_COST = data["backorder_cost_per_unit_per_period"]
D_real  = data["demand_realized"]
 
 
# Read 1a output
df_prod = pd.read_excel("output_assignment1a.xlsx", sheet_name="Production Plan", index_col=0)
df_cost = pd.read_excel("output_assignment1a.xlsx", sheet_name="Cost Summary",    index_col=0)
 
prod_e2801           = [float(df_prod.loc["E2801", f"W{t}"]) for t in periods]
setup_cost_1a        = float(df_cost.loc["TOTAL", "Setup Cost (EUR)"])
holding_cost_1a      = float(df_cost.loc["TOTAL", "Holding Cost (EUR)"])
holding_cost_e2801   = float(df_cost.loc["E2801", "Holding Cost (EUR)"])
holding_cost_other   = holding_cost_1a - holding_cost_e2801
 
 
# Simulate E2801 under realized demand
ending_inventory, ending_backorder = [], []
arrivals_list, holding_list, backorder_list, delivered_list = [], [], [], []
 
prev_inv, prev_bo = I0["E2801"], 0.0
 
for t in periods:
    t_order  = t - LT["E2801"]
    arrivals = prod_e2801[t_order - 1] if t_order >= 1 else 0.0
    arrivals_list.append(arrivals)
 
    available = prev_inv + arrivals
 
    if available >= prev_bo:
        avail_after_bo = available - prev_bo
        remaining_bo   = 0.0
    else:
        avail_after_bo = 0.0
        remaining_bo   = prev_bo - available
 
    demand = D_real[t - 1]
 
    if avail_after_bo >= demand:
        delivered   = demand
        inv_t       = avail_after_bo - demand
        new_bo      = 0.0
    else:
        delivered   = avail_after_bo
        inv_t       = 0.0
        new_bo      = demand - avail_after_bo
 
    bo_t = remaining_bo + new_bo
 
    ending_inventory.append(inv_t)
    ending_backorder.append(bo_t)
    holding_list.append(inv_t * HC["E2801"])
    backorder_list.append(bo_t * BO_COST)
    delivered_list.append(delivered)
 
    prev_inv, prev_bo = inv_t, bo_t
 
 
# KPIs
new_holding_e2801 = sum(holding_list)
total_holding     = holding_cost_other + new_holding_e2801
total_backorder   = sum(backorder_list)
total_cost        = setup_cost_1a + total_holding + total_backorder
 
total_demand      = sum(D_real)
fill_rate         = sum(delivered_list) / total_demand
service_level     = sum(1 for b in ending_backorder if b == 0) / T
 
print(f"Setup Cost (unchanged)   : EUR {setup_cost_1a:,.2f}")
print(f"Holding Cost - other     : EUR {holding_cost_other:,.2f}")
print(f"Holding Cost - E2801     : EUR {new_holding_e2801:,.2f}")
print(f"Total Holding Cost       : EUR {total_holding:,.2f}")
print(f"Backorder Cost           : EUR {total_backorder:,.2f}")
print(f"Total Cost               : EUR {total_cost:,.2f}")
print(f"Service Level            : {service_level:.4f}  |  Fill Rate: {fill_rate:.4f}")
print(f"Total Backorder Units    : {sum(ending_backorder):,.1f}")
 
 
# Excel output
df_summary = pd.DataFrame({
    "Metric": [
        "demand_type", "setup_cost", "holding_cost_other", "holding_cost_E2801",
        "total_holding_cost", "backorder_cost", "total_cost",
        "service_level", "fill_rate", "total_backorder_units",
    ],
    "Value": [
        "realized", round(setup_cost_1a, 2), round(holding_cost_other, 2),
        round(new_holding_e2801, 2), round(total_holding, 2),
        round(total_backorder, 2), round(total_cost, 2),
        round(service_level, 4), round(fill_rate, 4), round(sum(ending_backorder), 1),
    ],
})
 
df_e2801 = pd.DataFrame({
    "Week":                       periods,
    "Realized Demand":            D_real,
    "Production Released in 1A":  prod_e2801,
    "Arrivals":                   arrivals_list,
    "Ending Inventory E2801":     ending_inventory,
    "Ending Backorder E2801":     ending_backorder,
    "Holding Cost E2801 (EUR)":   holding_list,
    "Backorder Cost (EUR)":       backorder_list,
    "Units Delivered On Time":    delivered_list,
})
 
with pd.ExcelWriter("output_assignment1b.xlsx", engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary",          index=False)
    df_e2801.to_excel(writer,   sheet_name="E2801 Simulation", index=False)
 
print("\nOutput written to: output_assignment1b.xlsx\n")