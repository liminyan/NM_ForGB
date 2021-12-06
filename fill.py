import netCDF4 as nc
import numpy as np
import DataProcess as DP


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
	print('var_phs : ',var_phs,phs.shape)

	tar = 'pt'
	pt = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		pt.append(data)

	pt=np.array(pt)
	var_pt = np.var(pt,ddof=1)
	print('var_pt : ',var_pt,pt.shape)

	tar = 'u'
	u = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		u.append(data)

	u=np.array(u)
	var_u = np.var(u,ddof=1)
	print('var_u : ',var_u,u.shape)


	tar = 'v'
	v = []
	for lev in range(32):
		teg = 'scale 3.6' + tar + str(lev)+str(div)
		path = 'predic'+'/'+teg+'.npy'
		data = np.load(path)
		v.append(data)

	v=np.array(v)
	var_v = np.var(v,ddof=1)
	print('var_v : ',var_v,v.shape)

	return phs[1],pt[:,1],u[:,1],v[:,1]

phs,pt,u,v =  predic()
print(phs.shape)
print(pt.shape)
print(u.shape)
print(v.shape)
dataset = nc.Dataset('test.nc','a')
print(dataset.variables['phs'].shape)
print(dataset.variables['pt'].shape)
print(dataset.variables['u'].shape)
print(dataset.variables['v'].shape)


print(np.sum(dataset.variables['phs'][0] - phs))
print(np.sum(dataset.variables['pt'][0] - pt))
print(np.sum(dataset.variables['u'][0] - u))
print(np.sum(dataset.variables['v'][0] - v[:,:179,:]))

dataset.variables['phs'][0] = phs
dataset.variables['pt'][0] = pt
dataset.variables['u'][0] = u
dataset.variables['v'][0] = v[:,:179,:]

print(np.sum(dataset.variables['phs'][0] - phs))
print(np.sum(dataset.variables['pt'][0] - pt))
print(np.sum(dataset.variables['u'][0] - u))
print(np.sum(dataset.variables['v'][0] - v[:,:179,:]))

# dataset.variables['phs'] = np.zeros(1,180,360)
# dataset.variables['pt'] = np.zeros(1,32,180,360)
# dataset.variables['u'] = np.zeros(1,32,180,360)
# dataset.variables['v'] = np.zeros(1,32,179,360)
# print(dataset.variables['phs'].shape)
# dataset.close()

# print(dataset.variables['pt'].shape)
# print(dataset.variables['u'].shape)
# print(dataset.variables['v'].shape)



