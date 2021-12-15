import sys
import os
import mpi4py.MPI as MPI
import time


tar_l = ['phs','pt','u','v']
lev_l ={}
lev_l['phs'] = 1
lev_l['pt'] = 32
lev_l['u'] = 32
lev_l['v'] = 32
div = 1
p = 20

reset = 0 

for key in range(0,97):
	if key == 0:
		tar = tar_l[0]
		lev = 0
	else:
		tar = tar_l[int((key - 1) / 32 + 1)]
		lev = (key - 1) % 32
	 
	cmd = 'srun -n '+ str(p) +' /ddnstor/xuewei/3rdParty/anaconda3/bin/python3.7 -W ignore main.py'+ ' ' + str(tar) + ' ' + str(lev) + ' ' + str(div)
	switch = "sed -i ' 4c " + cmd +"'" +  ' run.sh'
	os.system(switch)

	cmd = '#SBATCH -J ' + str(tar) + '_' + str(lev) + '_' + str(div)
	switch = "sed -i ' 2c " + cmd +"'" +  ' run.sh'
	os.system(switch)

	if reset == 1:
		f = open(str(key)+'.txt','w')
		f.write('')
		f.close()
	else:
		f = open(str(key)+'.txt','r')
		l = len(f.readline())
		f.close()
		if l <= 1:
			cmd = 'sbatch '+ '--error='+str(key)+'.txt' +  ' -p cnCPU  ' + ' -n  '+ str(p) +' run.sh'
			os.system(cmd)
			print(cmd)



