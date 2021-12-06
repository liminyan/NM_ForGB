import netCDF4 as nc
import numpy as np
import DataProcess as DP


def get_org():
	print('org')
	train_data_path = '/home/xuewei/ddn/share/svr_data/'
	file = DP.get_nc_from_file(train_data_path)
	file.sort()
	tar_l = ['phs','pt','u','v']


	phs = []
	pt = []
	u = []
	v = []
	for x in range(1,17):
		# print(train_data_path+file[x])
		dataset = nc.Dataset(train_data_path+file[x])
		# print(dataset.variables['phs'].shape)
		# print(dataset.variables['pt'].shape)
		# print(dataset.variables['u'].shape)
		# print(dataset.variables['v'].shape)
		phs.append(dataset.variables['phs'][2])
		pt.append(dataset.variables['pt'][2])
		u.append(dataset.variables['u'][2])
		v.append(dataset.variables['v'][2])


	phs = np.array(phs)
	pt = np.array(pt)
	u = np.array(u)
	v = np.array(v)
	var_phs = np.var(phs,ddof=1)
	var_pt = np.var(pt,ddof=1)
	var_u = np.var(u,ddof=1)
	var_v = np.var(v,ddof=1)

	print('var_phs : ',var_phs)
	print('var_pt : ',var_pt)
	print('var_u : ',var_u)
	print('var_v : ',var_v)


def predic():
	print('predic')
	div = 1
	u = []
	v = []

	tar = 'phs'
	lev = 0
	teg = 'scale 3.6' + tar + str(lev)+str(div)
	path = 'predic'+'/'+teg+'.npy'
	phs = np.load(path)
	var_phs = np.var(phs,ddof=1)
	print('var_phs : ',var_phs)

	tar = 'pt'
	pt = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		pt.append(data)

	pt=np.array(pt)
	var_pt = np.var(pt,ddof=1)
	print('var_pt : ',var_pt)

	tar = 'u'
	u = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		u.append(data)

	u=np.array(u)
	var_u = np.var(u,ddof=1)
	print('var_u : ',var_u)


	tar = 'v'
	v = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		v.append(data)

	v=np.array(v)
	var_v = np.var(v,ddof=1)
	print('var_v : ',var_v)

# get_org()
# predic()

train_data_path = '/home/xuewei/ddn/share/svr_data/'
file = DP.get_nc_from_file(train_data_path)
file.sort()

print(file)