"""
Problem 1, 2 and solution for workshop
https://www.youtube.com/playlist?list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL
"""
import casadi as c
import numpy as np
from math import sin, pi

print(
    "########################\n\
    #       Problem 1      #\n\
    ########################")

print("Find minimum of x^2 - 6*x + 13")
x = c.SX.sym('w')
obj = x ** 2 - 6 * x + 13
g = []
# there are no parameters in this optimization prob
P = []
opt_variables = x
nlp_prob = {
    'f': obj,
    'x': opt_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        # Ipopt (interial Point Optimizer) : open source software package for large-scale nonlinear optimization.
        # this can be used to solve general NLPs
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = c.nlpsol('solver', 'ipopt', nlp_prob, opts)

# in this step,
#  min_x f(x,p)

# now we have to define constraints
# s.t x_min <= x <= x_max
#     g_min <= g(x,p) <= g_max

# unconstrained optimization
lbx = c.DM.zeros((1, 1))
ubx = c.DM.zeros((1, 1))
lbx[0, :] = -float('inf')
ubx[0, :] = float('inf')

lbg = c.DM.zeros((1, 1))
ubg = c.DM.zeros((1, 1))
lbg[0, :] = -float('inf')
ubg[0, :] = float('inf')
p=[]
# initialization of the optimization variable
x0 = -0.5
sol = solver(x0=-0.5, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
print(sol)
print(f"solution is x = {sol['x']}, minimum of obj is {sol['f']}")

####################################################################################

print(
    "########################\n\
    #       Problem 2      #\n\
    ########################")

print("Find minimum of exp(0.2*x)*sin(x)")
x = c.SX.sym('w')

####### CHANGE HERE #######
obj = c.exp(0.2 * x) * c.sin(x)

print(obj)

g = []
# there are no parameters in this optimization prob
p = []
opt_variables = x
nlp_prob = {
    'f': obj,
    'x': opt_variables,
    'g': g,
    'p': p
}

opts = {
    'ipopt': {
        # Ipopt (interial Point Optimizer) : open source software package for large-scale nonlinear optimization.
        # this can be used to solve general NLPs
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = c.nlpsol('solver', 'ipopt', nlp_prob, opts)

# unconstrained optimization
lbx = c.DM.zeros((1, 1))
ubx = c.DM.zeros((1, 1))
########CHANGE HERE#######
lbx[0, :] = 0
ubx[0, :] = 4 * pi

lbg = c.DM.zeros((1, 1))
ubg = c.DM.zeros((1, 1))
# we dont have useful funtion g.
lbg[0, :] = -float('inf')
ubg[0, :] = float('inf')

# initialization of the optimization variable
# there is multiple local minima b/w lower and upper bound.
# so the solution can be differ depending on x0 
x0 = 1
sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
print(sol)
print(f"solution is x = {sol['x']}, minimum of obj is {sol['f']}")

####################################################################################
