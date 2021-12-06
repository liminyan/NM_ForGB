import numpy as np
import DataProcess as DP
import netCDF4 as nc
import Function as func
train_data_path = '/home/xuewei/ddn/share/svr_data/'
saved_data_path = '/share1/liminyan/save_data/'
test_file = '/home/xuewei/ddn/share/svr_data/mpas.360x180.for_ylm_06.l32.h0_000003.nc'
output_path = 'predic/'
file = DP.get_nc_from_file(train_data_path)
file.sort()
tar_l = ['phs','pt','u','v']

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

def merge(model1,model2,tar,lev,div):

	path1 = output_path +model1+ str(tar)+str(lev)+str(div) + '.npy'
	data1 = np.load(path1)

	path2 = output_path +model2+ str(tar)+str(lev)+str(div) + '.npy'
	data2 = np.load(path2)

	path3 = output_path +'org' + '.npy'
	data3 = np.load(path3)


	# dataset = nc.Dataset(test_file)
	# tar_res = dataset.variables[tar][int(div):17,int(lev)]
	# train_label= tar_res[:,[lev],:]

	print(data1.shape)
	print(data2.shape)
	print(data3.shape)

	rmse1 = np.sqrt(np.mean((data1-data3)**2))
	rmse2 = np.sqrt(np.mean((data2-data3)**2))

	get_mean(model1,data1)
	print(rmse1)
	print('/')
	get_mean(model2,data2)
	print(rmse2)
	print('/')
	get_mean('Org',data3)
	data_merge = (data1 + data2) / 2
	print('/')
	get_mean('merge',data_merge)
	rmse3 = np.sqrt(np.mean((data_merge-data3)**2))
	print(rmse3)
	test = []
	for x in range(16):
		merge=np.c_[data1[x],data2[x],data_merge[x],data3[x]]
		test.append(merge)
	func.draw(list(test) ,'test',teg ='merge')




merge(model1 = 'svr',model2 = 'line',tar='pt',lev='31',div='1')