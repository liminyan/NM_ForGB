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
import gc

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

def set_input_m(tar,model = '',lev = 32,div = 1,bias = "",model_save_path = '/share1/liminyan/model'):
	print('model:',model)
	s = time.time()
	if tar == 'phs':
		lev = 0

	if model == '0':
		path = model_save_path + '/'+ model +bias+tar+'_'+str(block_num) +'_'+str(lev)+'_0.npy'
	elif model == '1' :
		path = model_save_path + '/'+ model +bias+tar+'_'+str(block_num) +'_'+str(lev)+'_1.npy'
	else:
		path = model_save_path + '/'+ model +bias+tar+'_'+str(block_num) +'_'+str(lev)+'.npy'

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


def train_model(tar,lev,div,batch_epc = 1,bias = '',div_num = 8, ord_ = 0):
	
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	tar_l = ['phs','pt','u','v']

	path = set_input_m(tar,
		model = str(ord_),lev = lev,div = div,
		model_save_path = saved_model_path,
		bias = bias)
	Tr = TrainData.TrainData(ensemble_nums= 32 * 3 + 1, lats=180, lons=360)
	print(path)

	num = 10
	svr = Model.SupportVectorRegression()
	proxy = PostProcessor.PostProcessor(Tr, svr)

	teg = bias + str(tar)+str(lev)+str(div)

	if comm_rank == 0:
		test = []

	for x in range(batch_epc):
		for ite in range(div_num):

			s = time.time()
			Train_x,Train_y, = DP.get_train_npy_from_nc_min_size(
				file,
				tar_l,
				train_data_path,
				batch_epc=x,
				lev = lev,
				labe = tar,
				save_path = saved_data_path,
				num= ite,
				div_num = div_num
				)

			# Train_x = np.array(Train_x).astype(np.float64)
			# Train_y = np.array(Train_y).astype(np.float64)
			print(Train_x.shape,Train_y.shape)

			Train_x = Train_x[ord_::2]
			Train_y = Train_y[ord_ + div::2]

			min_len =  min(Train_x.shape[0],Train_y.shape[0])

			Train_x = Train_x[:min_len]
			Train_y = Train_y[:min_len]
			
			Tr.loadData(from_path=False,
				x_train = Train_x,
				y_train = Train_y,
				x_test = Train_x[:num],
				y_test = Train_y[:num])

			Tr.show()
			# 	# proxy.load(path) #180*360 / size 
			# 	# 每个进程少部分模型，他自己的数据，
			e = time.time()
			if bias == 'svr':
				proxy.fit()
			elif bias == 'line':				
				proxy.partial_fit('line')
			else: 
				# bias == 'mlp':				
				proxy.partial_fit('mlp')
			print('rank',comm_rank,'total time',e-s)
			print(teg)
			del Train_x
			del	Train_y
			gc.collect()

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



def pre_model(tar,lev,div,batch_epc = 1,bias = '',proxy = None,div_num = 8, ord_ = 0):
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	tar_l = ['phs','pt','u','v']
	num = 16

	path = set_input_m(
		tar,
		model = str(ord_),
		lev = lev,
		div = div,
		bias = bias,
		model_save_path = saved_model_path)
	Tr = TrainData.TrainData(ensemble_nums= 32 * 3 + 1, lats=180, lons=360)
	# print(path)
	# return
	ord_ = (ord_ + 1) % 2

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
		s = time.time()
		Train_x,Train_y, = DP.get_train_npy_from_nc_min_size(
			file,
			tar_l,
			train_data_path,
			batch_epc=batch_epc,
			lev = lev,
			labe = tar,
			save_path = saved_data_path,
			num = 1,
			div_num = div_num)

		# Train_x = np.array(Train_x).astype(np.float64)
		# Train_y = np.array(Train_y).astype(np.float64)
		print(Train_x.shape,Train_y.shape)
		# print(test1[0::2])
		# print(test1[1::2])

		Train_x = Train_x[ord_+0::2]
		Train_y = Train_y[ord_+div::2]

		Tr.loadData(from_path=False,
			x_test = Train_x[:num],
			y_test = Train_y[:num])

		print(Train_x.shape,Train_y.shape)
		proxy.data = Tr
		print(proxy.data.x_test.shape,proxy.data.y_test.shape)

		# proxy.model.line_num = get_line(tar,lev)
		proxy.predict()
		e = time.time()
		print('rank',comm_rank,'total time',e-s)
		if comm_rank == 0:
			print(proxy.calRMSE())
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
		gc.collect()

	if comm_rank == 0:
		s = time.time()
		func.draw(list(test) ,'test',teg = teg+'again')
		e = time.time()
		print('draw time',e - s)
	# proxy.save(path)

data_path = '/home/xuewei/ddn/share/svr_data/svr_data_1month/'
data_path = '/home/xuewei/ddn/share/svr_data/svr_data_100km_12month_ACGrid/'

train_data_path = data_path
saved_data_path = '/share1/liminyan/save_data/'
saved_model_path = data_path +'model/'
output_path = 'Data/'
block_num = 1
file = DP.get_nc_from_file(train_data_path)
file.sort()


if __name__ == '__main__':

	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	begin = time.time()
	tar = 'pt'
	lev = 1 #[0 , 31]
	div = 2

	batch_epc = 16*12
	if len(sys.argv) > 3:
		tar = sys.argv[1]
		lev = int(sys.argv[2])
		div = int(sys.argv[3])
	model = 'line'
	print('...:',model)
	proxy = None
	# proxy = train_model(tar,lev,div,batch_epc = batch_epc,bias = model)
	if comm_rank == 0:
		print('end!')

	pre_model(tar,lev,div,bias = model,proxy = proxy)

	if comm_rank == 0:
		e = time.time()
		print(batch_epc,'draw time',e - begin)







