import numpy as np
import matplotlib.pyplot as plt
import csv



dt = 1e-3
# Caminho inteiro na minha maquina
fname = "C:/Users/WINDOWS 10/Documents/Projetos/Git_GISIS_training/GISIS_training/signals_n_systems/cmp_gather_5001x161_1000us.bin"
cmp = np.fromfile(fname, count = 805161, dtype = np.float32)
cmp = cmp.reshape((5001,161), order = "F")
cmps = cmp*(dt)


fonte = 5001
recp = 161
dx = 25
camadas = 4

#-------------------------
G= np.c_[np.ones(recp), (np.arange(recp) * dx)**2]
t0=np.zeros((camadas,recp))
#-------------------------


#identificação  de chegada de picos  em Dados sismicos (CMP)

for offset in range(recp):
    max_amplitude = np.max(cmps[:,offset])
    picks = np.where(cmps[:,offset] == max_amplitude)[0]
    dpicks = np.append([0], picks[1:] - picks[:-1]) 
    t0[:len(picks), offset] = np.delete(picks, np.where(dpicks == 1)) * dt
    
t0=t0*t0


def inverse(G, T):
    """
    Função para realizar inversão de modelo sísmico.
    
    Parâmetros:
        G: Matriz de projeto (recp x 2)
        T: Matriz de chegadas de ondas refletidas (camdas x recp)
    
    Retorna:
        velocidade de cada camadas e tempo 
    """
    GTG = np.dot(G.T, G)
    
    camadas = len(T)
    t0 = np.zeros(camadas)
    vrms = np.zeros(camadas)
    vint = np.zeros(camadas)
    prof = np.zeros(camadas)
    for i in range(camadas):
        GTd = np.dot(G.T, T[i, :])        
        m = np.linalg.solve(GTG, GTd)
        t0[i] = np.sqrt(m[0])
        vrms[i] = 1.0 / np.sqrt(m[1])
    
    return t0, vrms,m,vint

tt, vrms , m  , vint  = inverse(G, t0)
Prof = np.zeros(4)
vint[0] = vrms[0]

Prof[0] = 0.5*tt[0]*vint[0]
for i in range (1,4):
    vint[i] = np.sqrt(((vrms[i])**2*tt[i]-(vrms[i-1])**2*tt[i-1])/(tt[i]-tt[i-1]))
    Prof[i] = Prof[i-1] + 0.5*(tt[i]-tt[i-1])*vint[i]
    

# Parametros verdadeiros 

true_vint = np.array([1500, 1650, 2000, 3000, 4500])
true_depth = np.array([500, 1500, 2500, 3500])

z = np.arange(int((np.max(true_depth) + 3000)/dx))*dx

inv_model = vint[0] * np.ones_like(z)
true_model = true_vint[0] * np.ones_like(z) 
for i in range(1, len(true_vint)):        
    true_model[int(true_depth[i-1]/dx):] = true_vint[i]
    inv_model[int(Prof[i-1]/dx):] = vint[i] if i < len(vint) else np.nan

 
# Plot
    
fig , ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 9))

ax[0].imshow(cmps, aspect = "auto", cmap = "Greys")

for i in range(camadas):
    ax[0].plot(np.arange(recp), np.sqrt(tt[i]**2 + (np.arange(recp)*dx)**2/vrms[i]**2)/dt)

ax[0].plot(np.zeros(len(tt)), tt/dt, "ro")
ax[0].set_xticks(np.linspace(0, recp, 5))
ax[0].set_xticklabels(np.linspace(0, recp-1, 5)*dx)

ax[0].set_yticks(np.linspace(0, fonte, 11))
ax[0].set_yticklabels(np.linspace(0, fonte-1, 11)*dt)

ax[0].set_title("CMP Gather", fontsize = 18)
ax[0].set_xlabel("x = Offset [m]", fontsize = 15)
ax[0].set_ylabel("t = TWT [s]", fontsize = 15)

ax[1].plot(true_model, z)
ax[1].plot(inv_model, z)

ax[1].set_title("Estimated model", fontsize = 18)
ax[1].set_xlabel("velocities [m/s]", fontsize = 15)
ax[1].set_ylabel("Prof[m]", fontsize = 15)

ax[1].set_ylim([0, z[-1]])
ax[1].invert_yaxis()

fig.tight_layout()
plt.show()