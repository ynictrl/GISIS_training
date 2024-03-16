import numpy as np
from function import polynomial_function, noise, least_squares 

# --------------------- polynomial_function ---------------------

# parameters = [11, 24, 35]
# x = 2
# polynom = 0

# for k, p in enumerate(parameters): # (k == index) (p == value)		
# 	polynom += p*x**k
# 	print(f'{polynom:3} + {p}*{x}**{k} = {polynom}')
	
# 	# (0*x**parameters[0])
# 	# (1*x**parameters[1])
# 	# (2*x**parameters[2])
	
# ---------------------------- noise ----------------------------

# data = [3, 2, 4, 6]
# noise_amplitude = 0.2

# res = data + noise_amplitude*(0.5 - np.random.rand(len(data)))

# print(len(data), np.random.rand(len(data)), res)

# # print(np.random.rand(4)) # lista aleatoria com numwro de index determidado

# ------------------------ least_squares ------------------------

x = np.linspace(0, 2, 1000, dtype=float)
order = len(np.array([0.50, 0.20, -0.10, 0.30]))

y_true = polynomial_function(np.array([0.50, 0.20, -0.10, 0.30]), x)
y_noise = noise(y_true, 0.1)
y_noise[500:700] *= 0.9

d = y_noise

G = np.zeros((len(d), order))
	
for k in range(order): 
	G[:,k] = x**k

GTG = G.T @ G
GTd = G.T @ d

print(GTG, '\n', GTd, '\n', np.linalg.solve(GTG, GTd)) 
