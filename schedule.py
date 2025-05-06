# schedule.py

import pulp

model = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

a1 = pulp.LpVariable('a1', lowBound=0, upBound=4, cat=pulp.LpInteger)
a2 = pulp.LpVariable('a2', lowBound=0, upBound=3, cat=pulp.LpInteger)
b = pulp.LpVariable('b', lowBound=0, upBound=16, cat=pulp.LpInteger)
c1 = pulp.LpVariable('c1', lowBound=0, upBound=2, cat=pulp.LpInteger)
c2 = pulp.LpVariable('c2', lowBound=0, upBound=2, cat=pulp.LpInteger)
c3 = pulp.LpVariable('c3', lowBound=0, upBound=3, cat=pulp.LpInteger)


model += 5000*(a1 + a2) + 8000*b + 3000*(c1 + c2 + c3), "Total_Cost"

model += 3*(a1 + a2) + 5*b + 2*(c1 + c2 + c3) >= 22.28
model += 5000*(a1 + a2) + 8000*b + 3000*(c1 + c2 + c3) <= 1e5

model.solve()

print("Status:", pulp.LpStatus[model.status])
for v in model.variables():
    print(v.name, "=", v.varValue)

print("Minimum Cost =", pulp.value(model.objective))