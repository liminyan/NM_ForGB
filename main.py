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
import test
import warnings
warnings.filterwarnings("ignore")

def get_mean(bias,results):
	mean = results[0]
	for x in range(1,results.shape[0]):
		mean += results[x]
	mean /= results.shape[0]

	res = []
	for x in range(0,results.shape[0]):
		res.append((results[x]-mean)**2)

	res  = np.array(res)
	print(bias,(np.mean(res)))

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

def show_res(tar,block_num,lev,results,num,model = None):

	if model == 'self':
		path = tar + 'results_'+str(block_num)+'_'+str(lev)+'_selfV.npy'
	elif model == 'merge_uv' :
		path = tar + 'results_'+str(block_num)+'_'+str(lev)+'_mergeuv.npy'
	else:
		path = tar + 'results_'+str(block_num)+'_'+str(lev)+'.npy'

	np.save(path,results)
	results = np.load(path)

	func.draw(list(results[0:num]),tar)
	np.save(tar+'sub_res.npy',results[0:num])

def train_model(file,tar,lev,div,batch_epc = 16,bias = '',num = 2):
	
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

	for x in range(batch_epc):
		for ite in range(2):
			Train_x,Train_y, = DP.get_train_npy_from_nc_min_size(
				file,
				tar_l,
				train_data_path,
				batch_epc=x,
				lev = lev,
				labe = tar,
				save_path = saved_data_path,
				num= ite
				)

			Train_x = np.array(Train_x).astype(np.float64)
			Train_y = np.array(Train_y).astype(np.float64)

			Train_x = Train_x[:-div]
			Train_y = Train_y[div:]

			Tr.loadData(from_path=False,
				x_train = Train_x,
				y_train = Train_y,
				x_test = Train_x[:num],
				y_test = Train_y[:num])

			e = time.time()
			if bias == 'svr':
				proxy.fit()
			elif bias == 'line':				
				proxy.partial_fit('line')
			else: 
				proxy.partial_fit('mlp')

			print('rank',comm_rank,'total time',e-begin)
			del Train_x
			del	Train_y

	if comm_rank == 0:
		s = time.time()
		func.draw(list(test) ,'test',teg = teg )
		e = time.time()
		print('draw time',e - s)
	proxy.save(path)

	return proxy

def get_line(tar,lev):
	tar_l = ['phs','pt','u','v']

	if tar == 'phs':
		return 0

	if tar == 'pt':
		return 1 + lev

	if tar == 'u':
		return 1 + 32 + lev

	if tar == 'v':
		return 1 + 32 + 32 + lev

def pre_model(tar,lev,div,batch_epc = 1,bias = '',proxy = None):
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	tar_l = ['phs','pt','u','v']
	num = 16

	path = set_input_m(
		tar,
		model = '',
		lev = lev,
		div = div,
		num = num,
		bias = bias,
		model_save_path = saved_model_path)
	Tr = TrainData.TrainData(ensemble_nums= 32 * 3 + 1, lats=180, lons=360)
	# print(path)
	# return
	if proxy == None:

		svr = Model.SupportVectorRegression()
		svr.tar = tar
		proxy = PostProcessor.PostProcessor(Tr, svr)
		#################
		proxy.load(path)#
		################# 
	proxy.model.line_num = get_line(tar,lev)
	if comm_rank == 0:
		test = []

	teg = bias + str(tar)+str(lev)+str(div) 
	for x in range(batch_epc):
		Train_x,Train_y, = DP.get_train_npy_from_nc_min_size(
			file[1:],
			tar_l,
			train_data_path,
			batch_epc=batch_epc,
			lev = lev,
			labe = tar,
			save_path = saved_data_path,
			num = 1)

		Train_x = np.array(Train_x).astype(np.float64)
		Train_y = np.array(Train_y).astype(np.float64)

		Train_x = Train_x[:-div]
		Train_y = Train_y[div:]

		Tr.loadData(from_path=False,
			x_test = Train_x[:num],
			y_test = Train_y[:num])
		print(Train_x.shape,Train_y.shape)
		proxy.data = Tr
		# proxy.model.line_num = get_line(tar,lev)
		proxy.predict()
		e = time.time()
		print('rank',comm_rank,'total time',e-begin)
		if comm_rank == 0:
			# print(proxy.calRMSE())
			print('sove data')
			results = proxy.getResults()
			label_y = Tr.y_test
			for x in range(num):
				merge=np.c_[results[x],label_y[x]]
				test.append(merge)
			
			np.save('predic'+'/'+teg+'.npy',results[:num])
			np.save('predic'+'/'+'org'+'.npy',label_y[:num])

			get_mean('Prc',results[:num])
			get_mean('Org',label_y[:num])

		del Train_x
		del	Train_y

	if comm_rank == 0:
		s = time.time()
		func.draw(list(test) ,'test',teg = teg+'again')
		e = time.time()
		print('draw time',e - s)



train_data_path = '/home/xuewei/ddn/share/svr_data/'
saved_data_path = '/share1/liminyan/save_data/'
saved_model_path = '/home/xuewei/ddn/share/svr_data/model/'
output_path = 'Data/'
block_num = 1
file = DP.get_nc_from_file(train_data_path)
file.sort()
file = [x for x in file if '.nc' in x]

if __name__ == '__main__':

	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	begin = time.time()

	tar = 'pt'
	lev = 1 #[0 , 31]
	div = 2

	if len(sys.argv) > 3:
		tar = sys.argv[1]
		lev = int(sys.argv[2])
		div = int(sys.argv[3])
	model = 'mlp'
	print('...:',model)
	proxy = None
#	proxy = train_model(file,tar,lev,div,bias = model)
	# pre_model(tar,lev,div,bias = model,proxy = proxy)

	test.test_train(file,tar,lev,div,bias = model,saved_model_path = saved_model_path)









