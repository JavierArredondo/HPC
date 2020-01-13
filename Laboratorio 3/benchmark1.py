import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

verbose = True

def getEjecutionTime(N, xy, T, t):
	times = []
	for config in xy:
		x = config[0]
		y = config[1]
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

N = 2048
xy = [[16, 16], [32, 16], [32, 32]]
T = [300, 1000, 10000]
t = 300

times = []
for TT in T:
	timecuda = getEjecutionTime(N, xy, TT, t)
	times.append(timecuda)

col = ["b", "r", "g"]

for i in range(3):
	fig = plt.figure()
	ti = np.asarray([times[0][i], times[1][i], times[2][i]])
	plt.plot(ti, np.asarray(T), col[i])
	fig.suptitle(f'Tiempo de ejecución [s] - {xy[i][0]}x{xy[i][1]}')
	plt.xlabel('Tiempo')
	plt.ylabel('Número de iteraciones')
	fig.savefig('images/test{}.jpg'.format(f'_x{xy[i][0]}_y{xy[i][1]}_t{t}'))
