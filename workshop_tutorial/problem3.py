"""
Problem 3 and solution for workshop
https://www.youtube.com/playlist?list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL
"""
import casadi as ca
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

print(
    "########################\n\
    #       Problem 3      #\n\
    ########################")
"""
For the following set of data points, fit a straight line of the form
y = m*x + c

Here, we minimize the sum of the
squared errors, between the line and
the data points (Least squares).
"""
print("Find least square solution by optimizing m,c")

x = [0, 45, 90, 135, 180];
y = [667, 661, 757, 871, 1210];

plt.figure(1)
plt.scatter(x, y, label='Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
m = ca.SX.sym('m')  # Decision variable(slope)
c = ca.SX.sym('c')  # Decision variable(y - intersection)

obj = 0
for i in range(len(x)):
    obj = obj + (y[i] - (m * x[i] + c)) ** 2;
g = []  # Optimization constraints – empty(unconstrained)
P = []  # Optimization problem parameters – empty(no parameters used here)

OPT_variables = ca.vertcat(m, c)  # % Two decision variable
n_states = OPT_variables.size()[0]
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {
    'ipopt': {
        # Ipopt (interial Point Optimizer) : open source software package for large-scale nonlinear optimization.
        # this can be used to solve general NLPs
        'max_iter': 1000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# unconstrained optimization
lbx = ca.DM.zeros((1, 1))
ubx = ca.DM.zeros((1, 1))
lbx[0, :] = -float('inf')
ubx[0, :] = float('inf')

lbg = ca.DM.zeros((1, 1))
ubg = ca.DM.zeros((1, 1))
lbg[0, :] = -float('inf')
ubg[0, :] = float('inf')
p = []  # There are no parameters in this optimization problem
x0 = [0.5, 1]  # initialization of the optimization problem

sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
x_sol = sol['x']  # Get the solution
min_value = sol['f']  # Get the value function
print(f"solution is x = {x_sol}, minimum of obj is {min_value}")

x_line = list(range(0, 180))
m_sol = x_sol[0]
c_sol = x_sol[1]
y_line = [m_sol * i + c_sol for i in x_line]
plt.figure(1)
plt.plot(x_line, y_line, label='y = 2.88 x + 574')
plt.legend()
plt.show()

# visualize the objective function
# Function makes a function by sym variable.
# the input and output variable used in the function must be defined in advance as sym variables.
obj_fun = ca.Function('obj_fun', [m, c], [obj])

m_range = np.arange(-1, 6, 0.5)
c_range = np.arange(400, 800, 50)

[mm, cc] = np.meshgrid(m_range, c_range)
obj_plot_data = np.zeros_like(mm)
for n in range(mm.shape[0]):
    for k in range(mm.shape[1]):
        obj_plot_data[n, k] = obj_fun(mm[n, k], cc[n, k])
fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_surface(mm, cc, obj_plot_data);

ax.set_xlabel('(m)')
ax.set_ylabel('(c)')
ax.set_zlabel(r'$(\phi)$')
plt.show()
print(np.min(obj_plot_data))
