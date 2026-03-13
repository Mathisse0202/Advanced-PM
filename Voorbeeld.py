import gurobipy as gp
from gurobipy import GRB

#Parameters
totalFloorspace = 500

minimumModel1 = 20
minimumModel2 = 30

requiredSpaceModel1 = 5
requiredSpaceModel2 = 6

sellingPriceModel1 = 15
sellingPriceModel2 = 20

#Construct optimization model
model = gp.Model('APM_project_2026_example')
model.setParam('MIPGap', 0)

#Decision variables
orderModel1 = model.addVar(vtype=GRB.INTEGER, name="Order for model 1")
orderModel2 = model.addVar(vtype=GRB.INTEGER, name="Order for model 2")

#Objective function
model.setObjective(orderModel1*sellingPriceModel1 + orderModel2*sellingPriceModel2, GRB.MAXIMIZE)

#Constraints
model.addConstr((orderModel1 >= minimumModel1), name="Minimum order quantity for model 1")
model.addConstr((orderModel2 >= minimumModel2), name="Minimum order quantity for model 2")
model.addConstr((requiredSpaceModel1 * orderModel1 + requiredSpaceModel2 * orderModel2 <= totalFloorspace), name="Space constraint")
model.optimize()

print("Optimal order Model 1:", orderModel1.X)
print("Optimal order Model 2:", orderModel2.X)
print("Total revenue:", model.ObjVal)
