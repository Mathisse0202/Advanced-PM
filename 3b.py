from utils import *
import pandas as pd

data = load_data("input_data.json")
parts, periods, LT, BOM, parents = data["parts"], data["periods"], data["LT"], data["BOM"], data["parents"]
parts_ordered = ["B1401", "B2302", "B3201", "B4702", "V1501", "V6302", "E2801"]

# Solve 3a on forecast
m, p, q, y, _ = build_base_model(data, data["D_fcst"], "Assignment_3a", with_backorders=False)
ox, oy = add_overtime_vars(m, data)
add_capacity_with_overtime(m, p, ox, oy, data)
set_overtime_objective(m, p, q, y, ox, oy, None, data, with_backorders=False)
m.optimize()
m.addConstr(sum(oy[t] for t in periods) <= 3155.0, name="fix_oy_total")
m.optimize()
p_star = {(i, t): int(round(p[i, t].X)) for i in parts for t in periods}

# Simulate on realized demand
q_sim, b_sim = {}, {t: 0 for t in periods}
for t in periods:
    for i in parts_ordered:
        q_prev   = data["I0"][i] if t == 1 else q_sim[(i, t - 1)]
        arriving = p_star[(i, t - LT[i])] if t - LT[i] >= 1 else 0
        ind_dem  = sum(BOM[par][i] * p_star[(par, t)] for par in parents.get(i, {}))

        if i == "E2801":
            total_needed  = (b_sim[t - 1] if t > 1 else 0) + data["D_real"][t - 1]
            served        = min(q_prev + arriving, total_needed)
            q_sim[(i, t)] = max(0, q_prev + arriving - served)
            b_sim[t]      = total_needed - served
        else:
            q_sim[(i, t)] = max(0, q_prev + arriving - ind_dem)

# Service metrics
service_level = 1 - sum(1 for t in periods if b_sim[t] > 0.5) / len(periods)
new_bo        = sum(max(0, b_sim[t] - (b_sim[t-1] if t > 1 else 0)) for t in periods)
fill_rate     = 1 - new_bo / sum(data["D_real"])

# Costs — overtime Y: sum minutes first, then convert to hours (matches model objective)
setup_cost      = sum(data["SC"][i] * round(y[i, t].X) for i in parts for t in periods)
holding_cost    = sum(data["HC"][i] * q_sim[(i, t)] for i in parts for t in periods)
backorder_cost  = sum(data["BO_COST"] * b_sim[t] for t in periods)
overtime_cost_x = sum(data["OT_COST_X"] * round(ox[t].X) for t in periods)
total_oy_min    = sum(oy[t].X for t in periods)
overtime_cost_y = data["OT_COST_Y"] * (total_oy_min / 60.0)
total_overtime  = overtime_cost_x + overtime_cost_y
total_cost      = setup_cost + holding_cost + total_overtime + backorder_cost

print(f"Service level: {service_level:.4f}  |  Fill rate: {fill_rate:.4f}")
print(f"Setup: {setup_cost:.2f}  Holding: {holding_cost:.2f}  OT_X: {overtime_cost_x:.2f}  OT_Y: {overtime_cost_y:.2f}  BO: {backorder_cost:.2f}")
print(f"TOTAL: {total_cost:.2f}")

# Excel output
df_summary = pd.DataFrame({
    "Metric": ["demand_type", "setup_cost", "holding_cost", "overtime_cost_X",
               "overtime_cost_Y", "total_overtime_cost", "backorder_cost",
               "total_cost", "service_level", "fill_rate"],
    "Value":  ["realized", round(setup_cost, 2), round(holding_cost, 2),
               round(overtime_cost_x, 2), round(overtime_cost_y, 2),
               round(total_overtime, 2), round(backorder_cost, 2),
               round(total_cost, 2), round(service_level, 4), round(fill_rate, 4)]
})

df_production = make_plan_df(p, parts, periods)
df_production.loc["Demand (real)"] = {f"W{t}": data["D_real"][t-1] for t in periods}

df_inventory = pd.DataFrame(
    {f"W{t}": {i: int(round(q_sim[(i, t)])) for i in parts_ordered} for t in periods}
)
df_inventory.loc["Backorders"] = {f"W{t}": int(round(b_sim[t])) for t in periods}

write_excel("Assignment_3b_output.xlsx", {
    "Summary":              df_summary,
    "Production Plan":      df_production,
    "Inventory Simulation": df_inventory,
})

# Na m.optimize(), print de ruwe waarden
for t in periods:
    if oy[t].X > 0.01:
        print(f"W{t}: oy={oy[t].X:.4f} min = {oy[t].X/60:.4f} h, cost={120*(oy[t].X/60):.4f}")
print(f"Som oy minuten: {sum(oy[t].X for t in periods):.4f}")
print(f"OT_Y cost exact: {120 * sum(oy[t].X for t in periods) / 60:.4f}")