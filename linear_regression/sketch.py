import numpy as np

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

data = [3, 2, 4, 6]
noise_amplitude = 0.2

res = data + noise_amplitude*(0.5 - np.random.rand(len(data)))

print(len(data), np.random.rand(len(data)), res)

print(np.random.rand(4)) # lista aleatoria com numwro de index determidado

# ------------------------ least_squares ------------------------

...