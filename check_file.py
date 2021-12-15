import DataProcess as DP
import netCDF4 as nc


lev_l = {}
lev_l['u'] = (2880, 32, 180, 360)
lev_l['v'] = (2880, 32, 179, 360)
lev_l['u_Agrid'] = (2880, 32, 180, 360)
lev_l['v_Agrid'] = (2880, 32, 180, 360)
lev_l['pt'] = (2880, 32, 180, 360)
lev_l['phs'] = (2880,180,360)

train_data_path = '/home/xuewei/ddn/share/svr_data/svr_data_100km_12month_ACGrid/'
file = DP.get_nc_from_file(train_data_path)
file.sort()
file = [_ for _ in file if '.nc' in _]
file = [train_data_path + _ for _ in file ]

print(file)
tar_l = ['phs','pt','u','v','u_Agrid','v_Agrid']

for x in range(len(file)):
	res = True
	dataset = nc.Dataset(file[x])
	for tar in tar_l:
		if dataset[tar].shape != lev_l[tar]: res = False
	print(x,res)
