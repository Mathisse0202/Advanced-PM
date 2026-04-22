from utils import *

# ===============================
# LOAD DATA
# ===============================
data = load_data("input_data.json")

parts   = data["parts"]
periods = data["periods"]
LT      = data["LT"]
BOM     = data["BOM"]
parents = data["parents"]

# ===============================
# STEP 1: SOLVE 3a (FORECAST)
# ===============================
m, p, q, y, _ = build_base_model(
    data,
    data["D_fcst"],
    model_name="Assignment_3a",
    with_backorders=False
)

ox, oy = add_overtime_vars(m, data)
add_capacity_with_overtime(m, p, ox, oy, data)
set_overtime_objective(m, p, q, y, ox, oy, None, data, with_backorders=False)

m.optimize()

# ===============================
# EXTRACT PLAN
# ===============================
p_star = {(i,t): int(round(p[i,t].X)) for i in parts for t in periods}

# ===============================
# STEP 2: SIMULATION (3b)
# ===============================
q_sim = {(i,t): 0 for i in parts for t in periods}
b_sim = {t: 0 for t in periods}

for i in parts:
    for t in periods:

        q_prev = data["I0"][i] if t == 1 else q_sim[(i,t-1)]

        # arrivals
        t_order = t - LT[i]
        arriving = p_star[(i,t_order)] if t_order >= 1 else 0

        # demand
        ext_demand = data["D_real"][t-1] if i == "E2801" else 0

        ind_demand = 0
        if i in parents:
            for parent in parents[i]:
                ind_demand += BOM[parent][i] * p_star[(parent,t)]

        total_demand = ext_demand + ind_demand

        if i == "E2801":
            available = q_prev + arriving
            served = min(available, total_demand)

            q_sim[(i,t)] = available - served

            unmet = total_demand - served
            b_prev = 0 if t == 1 else b_sim[t-1]
            b_sim[t] = b_prev + unmet
        else:
            q_sim[(i,t)] = q_prev + arriving - total_demand

# ===============================
# SERVICE METRICS
# ===============================
service_level = 1 - sum(1 for t in periods if b_sim[t] > 0) / len(periods)

total_demand = sum(data["D_real"])
total_backorders = sum(
    max(0, b_sim[t] - (b_sim[t-1] if t > 1 else 0))
    for t in periods
)

fill_rate = 1 - total_backorders / total_demand

# ===============================
# COSTS
# ===============================
setup_cost = sum(data["SC"][i] * y[i,t].X for i in parts for t in periods)

holding_cost = sum(
    data["HC"][i] * q_sim[(i,t)]
    for i in parts for t in periods
)

backorder_cost = sum(data["BO_COST"] * b_sim[t] for t in periods)

# ⚠️ OVERTIME = UIT 3a!
overtime_cost_x = sum(data["OT_COST_X"] * ox[t].X for t in periods)
overtime_cost_y = sum(data["OT_COST_Y"] * (oy[t].X / 60) for t in periods)
total_overtime_cost = overtime_cost_x + overtime_cost_y

total_cost = setup_cost + holding_cost + total_overtime_cost + backorder_cost

# ===============================
# PRINT
# ===============================
print("\n" + "="*60)
print("ASSIGNMENT 3B RESULTS (CORRECT)")
print("="*60)
print(f"Service level:        {service_level:.4f}")
print(f"Fill rate:            {fill_rate:.4f}")
print("-"*60)
print(f"Setup cost:           {setup_cost:.2f}")
print(f"Holding cost:         {holding_cost:.2f}")
print(f"Overtime cost X:      {overtime_cost_x:.2f}")
print(f"Overtime cost Y:      {overtime_cost_y:.2f}")
print(f"Total overtime cost:  {total_overtime_cost:.2f}")
print(f"Backorder cost:       {backorder_cost:.2f}")
print("-"*60)
print(f"TOTAL COST:           {total_cost:.2f}")
print("="*60)