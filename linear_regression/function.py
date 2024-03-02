import numpy as np

# FUNÇÃO POLINOMIAL
def polynomial_function(parameters, x):
	
	polynom = np.zeros_like(x)
	
	for k, p in enumerate(parameters):		
		polynom += p*x**k	
	
	return polynom  

# RUÍDO
def noise(data, noise_amplitude):
	return data + noise_amplitude*(0.5 - np.random.rand(len(data)))

# MÉTODO DOS MÍNIMOS QUADRADOS
def least_squares(x, order, d):

	G = np.zeros((len(d), order))
	
	for k in range(order): 
		G[:,k] = x**k

	GTG = G.T @ G
	GTd = G.T @ d

	return np.linalg.solve(GTG, GTd) 