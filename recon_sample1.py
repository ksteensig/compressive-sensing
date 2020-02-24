import numpy as np
import pyunlocbox as ulb

m = 50
N = 100

x = [0]*100
x[10] = 1
x[50] = 0.5
x[80] = 1

x = np.array(x)

A = np.random.normal(size=(m,N))

y = np.dot(A, x)

f1 = ulb.functions.norm_l1()
f2 = ulb.functions.proj_b2(epsilon=1e-6, y=y, A=A, tight=False)
x0 = np.array([0]*100)
solver = ulb.solvers.douglas_rachford()

ret = ulb.solvers.solve([f1, f2], x0, solver, maxit=500)

print(ret['sol'])
