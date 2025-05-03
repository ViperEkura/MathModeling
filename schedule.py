import pulp

model = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

x1 = pulp.LpVariable('x1', lowBound=0, upBound=2, cat=pulp.LpInteger)
x2 = pulp.LpVariable('x2', lowBound=0, upBound=3, cat=pulp.LpInteger)
y = pulp.LpVariable('y', lowBound=0, upBound=16, cat=pulp.LpInteger)
z1 = pulp.LpVariable('z1', lowBound=0, upBound=2, cat=pulp.LpInteger)
z2 = pulp.LpVariable('z2', lowBound=0, upBound=2, cat=pulp.LpInteger)
z3 = pulp.LpVariable('z3', lowBound=0, upBound=3, cat=pulp.LpInteger)


model += 5000*(x1 + x2) + 8000*y + 3000*(z1 + z2 + z3), "Total_Cost"

# 约束条件
model += 3*(x1 + x2) + 5*y + 2*(z1 + z2 + z3) >= 22.12
model += 5000*(x1 + x2) + 8000*y + 3000*(z1 + z2 + z3) <= 1e5

model.solve()

print("Status:", pulp.LpStatus[model.status])
for v in model.variables():
    print(v.name, "=", v.varValue)

print("Minimum Cost =", pulp.value(model.objective))