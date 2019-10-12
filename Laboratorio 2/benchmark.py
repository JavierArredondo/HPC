import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

THREADS = 24
verbose = True


def getSeqTime(tries):
	accum = 0
	for j in range(1, tries + 1):
		res = os.popen("./bin/mandelbrot -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f output/salida.raw").read()
		res = res.split()
		start = int(res[0])
		end = int(res[1])
		dif = end - start
		accum+=dif
		if(verbose):
			print("Response: {}\n Start: {}\n End: {}\n Diff: {}\n Accum: {}".format(res, start, end, dif, accum))
	print("Mean time: {} mics\n".format(accum/tries))
	return accum/tries

def getMeanTime(tries):
	times = []
	for i in range(1, THREADS+1):
		print("Thread: {}".format(i))
		accum = 0
		for j in range(1, tries + 1):
			res = os.popen("./bin/mandelbrotp -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f output/salida.raw -t {}".format(i)).read()
			res = res.split()
			start = int(res[0])
			end = int(res[1])
			dif = end - start
			accum+=dif
			if(verbose):
				print("Response: {}\n Start: {}\n End: {}\n Diff: {}\n Accum: {}".format(res, start, end, dif, accum))
		print("Mean time: {} mics\n".format(accum/tries))
		times.append(accum/tries)
	return times


seqTime = getSeqTime(5)
parTimes = getMeanTime(5)




threads = np.asarray(range(1, THREADS+1))
speedUp = seqTime / np.asarray(parTimes)

fig = plt.figure()
plt.plot(threads, speedUp)
fig.suptitle('Speed Up')
plt.xlabel('Número de tareas')
plt.ylabel('S(n)')
fig.savefig('images/test{}.jpg'.format(datetime.datetime.now()))