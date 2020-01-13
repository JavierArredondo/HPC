import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

verbose = True

def getEjecutionTime(Ns, x, y, T, t):
	times = []
	for N in Ns:
		cmd = f'./bin/wave -N {N} -x {x} -y {y} -T {T} -f output/salida_N{N}_x{x}_y{y}_T{T}_t{t} -t {t}'
		res = os.popen(cmd).read()
		print(res)
		res = res.split()
		start = int(res[-2])
		end = int(res[-1])
		dif = (end - start)/1000000
		print(f'N={N} x={x} y={y} T={T} t={t} {dif}[seconds]\n')
		times.append(dif)
	return times

Ns = [256, 512, 1024, 2048, 4096]
x = 16
y = 16
T = 10000
t = 300


timecuda =getEjecutionTime(Ns, x, y, T, t)
timecuda = np.asarray(timecuda)
Ns = np.asarray(Ns)

fig = plt.figure()
plt.plot(timecuda, Ns, "g")
fig.suptitle(f'Tiempo de ejecución [s] - {T} iteraciones')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño de grilla')
fig.savefig('images/test{}.jpg'.format(f'_x{x}_y{y}_T{T}_t{t}'))
