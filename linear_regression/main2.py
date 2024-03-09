# importando funções de outro script
from function import polynomial_function, noise, least_squares 

# bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ configurações -------------------------

N_DADOS = 1500

p_true = np.array([0.50, 0.20, -0.10, 0.30])

x = np.linspace(0, 2, N_DADOS, dtype=float)

# -------------------------- modelagem ---------------------------

y_true = polynomial_function(p_true, x)

y_noise = noise(y_true, 0.1)

y_noise[500:700] *= 0.9

# -------------------------- inversão ----------------------------

p_calc = least_squares(x, len(p_true), y_noise)

# -------------------------- validação ---------------------------

y_function = polynomial_function(p_calc, x)

# -------------------------- resultados --------------------------

fig, ax = plt.subplots(num = "Parameters estimation", figsize = (10, 7))

ax.plot(x, y_noise,    '.k', label = f"Observed data") # dado observado
ax.plot(x, y_true,     '-r', label = f"p_true = {np.around(p_true, decimals= 2)}") # dado verdadeiro
ax.plot(x, y_function, '-b', label = f"p_calc = {np.around(p_calc, decimals= 2)}") # dado calculado

ax.set_xlabel("X", fontsize = 15)
ax.set_ylabel("Y", fontsize = 15)
ax.grid(color='k', linestyle='-', linewidth=0.3)
ax.legend(loc = "upper left", fontsize = 12)

fig.tight_layout()
plt.show()
