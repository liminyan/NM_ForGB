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

def test_train(file,tar = 'v',lev ='31',div = '1',batch_epc = 1,bias = '',num = 2,saved_model_path = saved_model_path):
	
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
	
	return proxy