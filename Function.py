import matplotlib.pyplot as plt
import numpy as np


def draw(pictures,name = 'test',teg = ""):
	plt.switch_backend('agg')
	
	le = (pictures[0]).shape[1]
	le = int(le/360)
	for x in range(len(pictures)):
		plt.imshow(np.flipud(pictures[x]))

		for line in range(le):
			plt.vlines(line*360, ymin=0, ymax=180)  

		plt.colorbar(fraction=0.015, pad=0.01)
		num = str(x)
		if x <10:
			num = '00'+num
		elif x <100:
			num = '0' +num

		plt.savefig(name+'/'+teg+num+'.png')
		plt.close()

