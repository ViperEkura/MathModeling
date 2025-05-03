import pulp

model = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)
x1 = pulp.LpVariable('x1', lowBound=0, upBound=16)
x2 = pulp.LpVariable('x2', lowBound=0, upBound=16)
x3 = pulp.LpVariable('x3', lowBound=0, upBound=16)


model += 5000*x1 + 8000*x2 + 3000*x3, "Z"

model += 3*x1 + 5*x2 + 2*x3 >= 22.636
model += 5000*x1 + 8000*x2 + 3000*x3 <= 1e5

model.solve()
print("状态:", pulp.LpStatus[model.status])
print("最优值 Z =", pulp.value(model.objective))
print("最优解: x1 =", x1.value(), "x2 =", x2.value(),"x3=" ,x3.value())