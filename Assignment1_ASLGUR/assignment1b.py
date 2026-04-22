"""
Assignment 1b - Impact of Realized Demand and Backordering on Production Plan

Logic:
- Reuse the production plan from Assignment 1a
- Keep setup costs unchanged
- Keep intermediate-component holding costs unchanged
- Recalculate only E2801 inventory under realized demand
- Add backorder cost
- Compute fill rate and service level
"""

import json
import pandas as pd


# =============================================================================
# 1. LOAD INPUT DATA
# =============================================================================

with open("input_data.json", "r") as f:
    data = json.load(f)

T = data["planning_horizon"]
periods = list(range(1, T + 1))

I0 = data["initial_inventory"]
LT = data["lead_times"]
BO_COST = data["backorder_cost_per_unit_per_period"]
D_real = data["demand_realized"]


# =============================================================================
# 2. READ ASSIGNMENT 1A OUTPUT
# =============================================================================

file_1a = "output_assignment1a.xlsx"

df_prod = pd.read_excel(file_1a, sheet_name="Production Plan", index_col=0)
df_cost = pd.read_excel(file_1a, sheet_name="Cost Summary", index_col=0)

# 1A production plan for the final product
prod_e2801 = [float(df_prod.loc["E2801", f"W{t}"]) for t in periods]

# Costs from 1A
setup_cost_1a = float(df_cost.loc["TOTAL", "Setup Cost (EUR)"])
holding_cost_total_1a = float(df_cost.loc["TOTAL", "Holding Cost (EUR)"])
holding_cost_e2801_1a = float(df_cost.loc["E2801", "Holding Cost (EUR)"])

# Holding cost of all other parts remains unchanged
holding_cost_other_parts = holding_cost_total_1a - holding_cost_e2801_1a


# =============================================================================
# 3. SIMULATE E2801 UNDER REALIZED DEMAND
# =============================================================================

initial_inventory = I0["E2801"]
lead_time_e2801 = LT["E2801"]
holding_cost_rate_e2801 = data["holding_costs"]["E2801"]

ending_inventory = []
ending_backorder = []
arrivals_list = []
holding_cost_e2801_list = []
backorder_cost_list = []
delivered_on_time_list = []

prev_inventory = initial_inventory
prev_backorder = 0.0

for t in periods:
    # Arrivals in period t depend on the 1A production plan and lead time
    t_order = t - lead_time_e2801
    arrivals = prod_e2801[t_order - 1] if t_order >= 1 else 0.0
    arrivals_list.append(arrivals)

    # Inventory balance with backorders
    # available = inventory carried over + arrivals
    available = prev_inventory + arrivals

    # Meet previous backorders first
    if available >= prev_backorder:
        available_after_old_bo = available - prev_backorder
        remaining_old_bo = 0.0
    else:
        available_after_old_bo = 0.0
        remaining_old_bo = prev_backorder - available

    # Meet current realized demand
    current_demand = D_real[t - 1]

    if available_after_old_bo >= current_demand:
        delivered_on_time = current_demand
        inventory_t = available_after_old_bo - current_demand
        new_backorder = 0.0
    else:
        delivered_on_time = available_after_old_bo
        inventory_t = 0.0
        new_backorder = current_demand - available_after_old_bo

    backorder_t = remaining_old_bo + new_backorder

    ending_inventory.append(inventory_t)
    ending_backorder.append(backorder_t)
    holding_cost_e2801_list.append(inventory_t * holding_cost_rate_e2801)
    backorder_cost_list.append(backorder_t * BO_COST)
    delivered_on_time_list.append(delivered_on_time)

    prev_inventory = inventory_t
    prev_backorder = backorder_t


# =============================================================================
# 4. KPI CALCULATIONS
# =============================================================================

new_holding_cost_e2801 = sum(holding_cost_e2801_list)
new_total_holding_cost = holding_cost_other_parts + new_holding_cost_e2801
total_backorder_cost = sum(backorder_cost_list)
new_total_cost = setup_cost_1a + new_total_holding_cost + total_backorder_cost

total_realized_demand = sum(D_real)
total_backorder_units = sum(ending_backorder)

# Fill rate = demand delivered on time / total realized demand
fill_rate = sum(delivered_on_time_list) / total_realized_demand

# Service level = fraction of periods without end-of-period backorder
periods_without_backorder = sum(1 for b in ending_backorder if b == 0)
service_level = periods_without_backorder / T


# =============================================================================
# 5. PRINT RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("ASSIGNMENT 1B - REALIZED DEMAND WITH BACKORDERS")
print("=" * 70)
print(f"Setup Cost (unchanged)          : EUR {setup_cost_1a:,.2f}")
print(f"Holding Cost - other parts      : EUR {holding_cost_other_parts:,.2f}")
print(f"Holding Cost - E2801 (updated)  : EUR {new_holding_cost_e2801:,.2f}")
print(f"Total Holding Cost              : EUR {new_total_holding_cost:,.2f}")
print(f"Backorder Cost                  : EUR {total_backorder_cost:,.2f}")
print(f"Total Cost                      : EUR {new_total_cost:,.2f}")
print(f"Service Level                   : {service_level:.4f}")
print(f"Fill Rate                       : {fill_rate:.4f}")
print(f"Total Backorder Units           : {total_backorder_units:,.1f}")
print("=" * 70)


# =============================================================================
# 6. WRITE OUTPUT TO EXCEL
# =============================================================================

df_summary = pd.DataFrame([
    {"Metric": "Setup Cost (EUR)", "Value": round(setup_cost_1a, 2)},
    {"Metric": "Holding Cost - Other Parts (EUR)", "Value": round(holding_cost_other_parts, 2)},
    {"Metric": "Holding Cost - E2801 (EUR)", "Value": round(new_holding_cost_e2801, 2)},
    {"Metric": "Total Holding Cost (EUR)", "Value": round(new_total_holding_cost, 2)},
    {"Metric": "Backorder Cost (EUR)", "Value": round(total_backorder_cost, 2)},
    {"Metric": "Total Cost (EUR)", "Value": round(new_total_cost, 2)},
    {"Metric": "Service Level", "Value": round(service_level, 4)},
    {"Metric": "Fill Rate", "Value": round(fill_rate, 4)},
    {"Metric": "Total Backorder Units", "Value": round(total_backorder_units, 1)},
])

df_e2801 = pd.DataFrame({
    "Week": periods,
    "Realized Demand": D_real,
    "Production Released in 1A": prod_e2801,
    "Arrivals": arrivals_list,
    "Ending Inventory E2801": ending_inventory,
    "Ending Backorder E2801": ending_backorder,
    "Holding Cost E2801 (EUR)": holding_cost_e2801_list,
    "Backorder Cost (EUR)": backorder_cost_list,
    "Units Delivered On Time": delivered_on_time_list,
})

output_file = "output_assignment1b.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_e2801.to_excel(writer, sheet_name="E2801 Simulation", index=False)

print(f"\nOutput written to: {output_file}\n")