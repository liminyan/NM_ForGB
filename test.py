import netCDF4 as nc
import numpy as np
import DataProcess as DP
import Function as func
import TrainData 
import Model
import PostProcessor
import mpi4py.MPI as MPI
import time
import pickle
import sys

def set_input_m(tar,model = None,lev = 32,div = 1,num = 16,bias = "",model_save_path = '/share1/liminyan/model'):
	print('model:',model)
	s = time.time()
	if tar == 'phs':
		lev = 0

	if model == 'self':
		path = model_save_path + '/'+bias+tar+'_'+str(block_num) +'_'+str(lev)+'_self.npy'
	elif model == 'merge_uv' :
		path = model_save_path + '/'+bias+tar+'_'+str(block_num) +'_'+str(lev)+'_mergeuv.npy'
	else:
		path = model_save_path + '/'+bias+tar+'_'+str(block_num) +'_'+str(lev)+'.npy'

	e = time.time()
	print('set_input time',e - s)
	return path

def test_train(file,tar = 'v',lev ='31',div = '1',batch_epc = 1,bias = '',num = 2):
	
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	tar_l = ['phs','pt','u','v']
	num = 10
	path = set_input_m(tar,
		model = '',lev = lev,div = div,num = num,
		model_save_path = saved_model_path,
		bias = bias)
	Tr = TrainData.TrainData(ensemble_nums= 32 * 3 + 1, lats=180, lons=360)
	print(path)

	svr = Model.SupportVectorRegression()
	proxy = PostProcessor.PostProcessor(Tr, svr)

	teg = bias + str(tar)+str(lev)+str(div)

	if comm_rank == 0:
		test = []

	# for x in range(batch_epc):
	# 	for ite in range(2):
	# 		Train_x,Train_y, = DP.get_train_npy_from_nc_min_size(
	# 			file,
	# 			tar_l,
	# 			train_data_path,
	# 			batch_epc=x,
	# 			lev = lev,
	# 			labe = tar,
	# 			save_path = saved_data_path,
	# 			num= ite
	# 			)

	# 		Train_x = np.array(Train_x).astype(np.float64)
	# 		Train_y = np.array(Train_y).astype(np.float64)

	# 		Train_x = Train_x[:-div]
	# 		Train_y = Train_y[div:]

	# 		Tr.loadData(from_path=False,
	# 			x_train = Train_x,
	# 			y_train = Train_y,
	# 			x_test = Train_x[:num],
	# 			y_test = Train_y[:num])

	# 		e = time.time()
	# 		if bias == 'svr':
	# 			proxy.fit()
	# 		elif bias == 'line':				
	# 			proxy.partial_fit('line')
	# 		else: 
	# 			proxy.partial_fit('mlp')
	# 		print('rank',comm_rank,'total time',e-begin)
	# 		del Train_x
	# 		del	Train_y

	# if comm_rank == 0:
	# 	s = time.time()
	# 	func.draw(list(test) ,'test',teg = teg )
	# 	e = time.time()
	# 	print('draw time',e - s)
	# proxy.save(path)

	return proxy