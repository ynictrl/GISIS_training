from function import polynomial_function, noise, least_squares # importando funções de outro script

# bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# VARIÁVEIS

z_true = 500
v_true = 3000

p_true = [(2*z_true/v_true)**2, (1/v_true)**2]

order = len(p_true) # 2

off_min = 50
off_max = 8050
off_num = 321

offset = np.linspace(off_min, off_max, off_num) # criar array com espaçamento igual entre os index (n min, n max, len)

noise_amp = 50e-3 #ruido
exact_data = np.sqrt(polynomial_function(p_true, offset**2))

dado_obs = noise(exact_data, noise_amp)

p_calc = least_squares(offset**2, order, dado_obs**2)

v_calc, z_calc = np.sqrt(1/p_calc[1]), 0.5*np.sqrt(p_calc[0]/p_calc[1])

dado_cal = np.sqrt(polynomial_function(p_calc, offset**2))

#print(dado_cal)

# GRÁFICO

fig, ax = plt.subplots(num = "CMP parameters estimation", figsize = (10,7))

ax.plot(offset, dado_obs, '_', label = f"velocity = {v_true:.1f} m/s, depth = {z_true:.1f} m") # dado observado
ax.plot(offset, dado_cal, '-', label = f"velocity = {v_calc:.1f} m/s, depth = {z_calc:.1f} m") # dado calculado

ax.set_xlabel("Offset [m]", fontsize = 15)
ax.set_ylabel("TWT [s]", fontsize = 15)
ax.grid(color='k', linestyle='-', linewidth=0.3)
ax.legend(loc = "upper right", fontsize = 12)

fig.tight_layout()
plt.show()

# NOTAS:
# ax.invert_yaxis() # inverter eixo y
# plt.grid(color='k', linestyle='-', linewidth=0.3)
# ENCONTRAR EM UMA ATIVIDADE DO JUPYTER GRAFICO!!