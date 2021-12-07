import os
import netCDF4 as nc
import numpy as np
from pathlib import Path
import time
import mpi4py.MPI as MPI

lev_l = {}
lev_l['u'] = 32
lev_l['v'] = 32
lev_l['pt'] = 32
lev_l['phs'] = 1

def typemap(dtype):
    try:
        return MPI.__TypeDict__[np.dtype(dtype).char]
    except AttributeError:
        return MPI._typedict[np.dtype(dtype).char]

def parallel_read_array(filename, axis=0, comm=None):

    # if no MPI, or only 1 MPI process, call np.load directly
    if comm is None or comm.size == 1:
        return np.load(filename)

    # open the file in read only mode
    fh = MPI.File.Open(comm, filename, amode=MPI.MODE_RDONLY)

    # read and check version of the npy file
    version = fm.read_magic(fh)
    fm._check_version(version)
    # get shape, order, dtype info of the array
    global_shape, fortran_order, dtype = fm._read_array_header(fh, version)
    # print(global_shape, fortran_order, dtype)

    if dtype.hasobject:
        # contain Python objects
        raise RuntimeError('Currently not support array that contains Python objects')
    if fortran_order:
        raise RuntimeError('Currently not support Fortran ordered array')

    local_shape = list(global_shape)
    axis_len = local_shape[axis]
    base = axis_len / comm.size
    rem = axis_len % comm.size

    part = (base * np.ones(comm.size, dtype=np.int) + (np.arange(comm.size) < rem)).astype(np.int)
    bound = np.cumsum(np.insert(part, 0, 0))
    local_shape[axis] = part[comm.rank] # shape of local array
    local_start = [0] * len(global_shape)
    local_start[axis] = bound[comm.rank] # start of local_array in global array
    local_shape = [int(x) for x in local_shape]
    # allocate space for local_array to hold data read from file
    local_array = np.empty(local_shape, dtype=dtype, order='C')
    pos = fh.Get_position()
    etype = typemap(dtype)
    # construct the filetype
    filetype = etype.Create_subarray(global_shape, local_shape, local_start, order=MPI.ORDER_C)
    filetype.Commit()
    # set the file view
    fh.Set_view(pos, etype, filetype, datarep='native')
    # collectively read the array from file
    fh.Read_all(local_array)
    # close the file
    fh.Close()

    return global_shape,local_array


def get_nc_from_file(train_data_path,check = True):
	file = os.listdir(train_data_path)

	if check:
		print('all_size',len(file))
		print(file[0])
		dataset = nc.Dataset(train_data_path+file[0])
		key = dataset.variables.keys()
		print(key)
		print(dataset['time'].shape)
		print('u.shape',dataset['u'].shape)
		print('v.shape',dataset['v'].shape)
		print('pt.shape',dataset['pt'].shape)
		print('phs.shape',dataset['phs'].shape)
	return file

def get_npy_from_nc(file,tar_list,train_data_path,from_file = False,save_path = 'Data/'):

	start = time.time()
	d = [[] for x in range(len(tar_list))]
	print(tar_list,len(d))
	
	for tar in range(len(tar_list)):

		lev_range = lev_l[tar_list[tar]]

		if lev_range == 1:
			data = []
			for x in range(len(file)):
				dataset = nc.Dataset(train_data_path+file[x])
				data.append(np.array(dataset[tar_list[tar]])[0,:])
				if x % 16 == 0 and x:
					tm = time.time() - start
					print(tar,':',x,'/',len(file),'time :',round(tm,2),'/',round(tm/x * len(file),2) )

				if x % 1440 == 0 and x:

					np.save(save_path + tar_list[tar]+'_'+str(0)+'_'+str(x)+'.npy',data)
					del data
					data = []
			continue

		atime = 0
		btime = 0
		ctime = 0
		for lev in range(31,lev_range):
			data = []
			for x in range(len(file)):

				s = time.time()
				dataset = nc.Dataset(train_data_path+file[x])
				endtime = time.time()
				atime += (endtime - s)

				s = time.time()
				cur = np.array(dataset[tar_list[tar]])[0,lev,:]
				endtime = time.time()
				btime += (endtime - s)

				s = time.time()
				data.append(cur)
				endtime = time.time()
				ctime += (endtime - s)

				if x % 16 == 0 and x:
					tm = time.time() - start
					print(tar,':',x,'/',len(file),'time :',round(tm,2),'/',round(tm/x * len(file),2) )

				if x % 1440 == 0 and x:

					# np.save(save_path + tar_list[tar]+'_'+str(lev)+'_'+str(x)+'.npy',data)
					del data
					data = []
		print('read',atime,'to array',btime,'append',ctime)
		exit(0)

	elapsed = (time.time() - start)
	print("Read time used:",round(elapsed,2))


def get_train_npy_from_nc(file,tar_list,train_data_path,from_file = False,save_path = 'Data/'):
	start = time.time()
	d = [[] for x in range(len(tar_list))]
	print(tar_list,len(d))
	input_data = []
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	max_size = 180 * 360
	per_size = int(max_size/comm_size)
	begin = comm_rank*per_size
	end = (comm_rank + 1)*per_size
	i_begin = int(begin/360)
	i_end = int(end/360)
	print(i_begin,i_end)
	train_data = []


	for x in range(min(len(file),50)):
		dataset = nc.Dataset(train_data_path+file[x])
		merge = []
		for tar in range(len(tar_list)):
			lev_range = lev_l[tar_list[tar]]
			if lev_range!=1:


				tar_res = dataset.variables[tar_list[tar]][0,:,i_begin:i_end]
			else:
				tar_res = dataset.variables[tar_list[tar]][:,i_begin:i_end]
				stand_rank = tar_res.shape[1]
			if tar_res.shape[1] != stand_rank :
				tmp = np.zeros((360))
				tar_res = np.insert(tar_res,stand_rank-1,values = tmp,axis = 1)
			merge.append(tar_res)
		# merge = np.r_[merge[0],merge[1],merge[2],merge[3]]
		# print(merge[0].shape)
		# print(merge[1].shape)
		# print(merge[2].shape)
		# print(merge[3].shape) exit
		train_data.append(merge)
	elapsed = (time.time() - start)
	print("Read time used:",round(elapsed,2))
	return train_data


def get_train_npy_from_nc_min_size(
	file,
	tar_list,
	train_data_path,
	batch_epc=1,
	lev = 0,
	labe = 'phs',
	from_file = False,
	save_path = 'Data/',
	num = 1):

	start = time.time()
	d = [[] for x in range(len(tar_list))]
	print(tar_list,len(d))
	input_data = []
	comm = MPI.COMM_WORLD
	comm_rank = comm.Get_rank()
	comm_size = comm.Get_size()
	# comm_size = 120
	max_size = 180 * 360
	per_size = int(max_size/comm_size)
	begin = comm_rank*per_size   
	end = (comm_rank + 1)*per_size 
	i_begin = int(begin/360)
	i_end = int(end/360)

	j_begin = int(begin%360)
	j_end = int(end%360)
	train_data = []
	train_label = []

	x = batch_epc -1
	# for x in range(batch_num * batch_epc, batch_num * (batch_epc + 1)):
	print('>>',train_data_path+file[x],'>>',file[x])
	dataset = nc.Dataset(train_data_path+file[x])
	merge = []
	for tar in range(len(tar_list)):
		print(dataset.variables[tar_list[tar]].shape)

		lev_range = lev_l[tar_list[tar]]
		half_num = int(dataset.variables[tar_list[tar]].shape[0]/2)
		# half_num = int(dataset.variables[tar_list[tar]].shape[0])
		# half_num = 50
		half_num_begin = num*half_num
		half_num_end = half_num_begin+half_num

		if lev_range == 1:
			tar_res = None

			if j_begin == 0 and j_end == 0:
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin:i_end]
				shape = mid.shape
				mid = mid.reshape(shape[0],1,shape[1]*shape[2])
				tar_res = mid

			if j_begin != 0 and j_end == 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin,j_begin:]
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin+1:i_end]
				shape = mid.shape 
				mid = mid.reshape(shape[0],1,shape[1]*shape[2])
				tar_res = np.c_[first,mid]

			if j_begin == 0 and j_end != 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin:i_end]
				shape = first.shape 

				first = first.reshape(shape[0],1,shape[1]*shape[2])
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_end,:j_end]

				mid = mid.reshape(shape[0],1,shape[1]*shape[2])
				tar_res = np.c_[first,mid]

			if j_begin != 0 and j_end != 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin,j_begin:]
				shape = first.shape 
				first = first.reshape(shape[0],1,shape[1]*shape[2])
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_begin+1:i_end]
				shape = mid.shape 
				mid = mid.reshape(shape[0],1,shape[1]*shape[2])
				last = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,i_end,:j_end]
				shape = last.shape 
				last = last.reshape(shape[0],1,shape[1]*shape[2])
				tar_res = np.c_[first,mid,last]
			stand_rank = tar_res.shape[2]
		else:
			# ds = dataset.variables[tar_list[tar]]
			# if ds.shape[1] != stand_rank :
			# tmp = np.zeros((360))
			# tar_res = np.insert(tar_res,stand_rank-1,values = tmp,axis = 1)
			if j_begin == 0 and j_end == 0:
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin:i_end]
				shape = mid.shape
				mid = mid.reshape(shape[0],shape[1],shape[2]*shape[3])
				tar_res = mid

			if j_begin != 0 and j_end == 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin,j_begin:]
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin+1:i_end]
				shape = mid.shape 
				mid = mid.reshape(shape[0],shape[1],shape[2]*shape[3])
				tar_res = np.c_[first,mid]

			if j_begin == 0 and j_end != 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin:i_end]
				shape = first.shape 
				first = first.reshape(shape[0],shape[1],shape[2]*shape[3])
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_end,:j_end]
				tar_res = np.c_[first,mid]

			if j_begin != 0 and j_end != 0:
				first = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin,j_begin:]
				shape = first.shape 
				first = first.reshape(shape[0],shape[1],shape[2]*shape[3])
				mid = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_begin+1:i_end]
				shape = mid.shape 
				mid = mid.reshape(shape[0],shape[1],shape[2]*shape[3])
				last = dataset.variables[tar_list[tar]][half_num_begin:half_num_end,:,i_end,:j_end]
				shape = last.shape 
				last = last.reshape(shape[0],shape[1],shape[2]*shape[3])
				tar_res = np.c_[first,mid,last]

			if stand_rank != tar_res.shape[2]:
				err = stand_rank - tar_res.shape[2]
				tmp = np.zeros((tar_res.shape[0],tar_res.shape[1],abs(err)))
				tar_res = np.c_[tar_res,tmp]
				# print(stand_rank,tmp.shape,tar_res.shape)

		if labe ==  tar_list[tar] and labe == 'phs':	
			train_label = tar_res

		if labe ==  tar_list[tar] and labe != 'phs':	
			train_label= tar_res[:,[lev],:]
			shape = train_label.shape 
			# train_label= train_label.reshape(shape[0],1,shape[1])

		merge.append(tar_res)

	print(merge[0].shape,merge[1].shape,merge[2].shape,merge[3].shape)
	# merge = np.r_[merge[0],merge[1],merge[2],merge[3]]
	train_data = np.concatenate([merge[0],merge[1],merge[2],merge[3]],axis = 1)
	# train_label = np.array(train_label)
	print(labe,'train',train_data.shape)
	print(labe,'labe',train_label.shape)
	elapsed = (time.time() - start)
	print("Read time used:",round(elapsed,2))
	return train_data,train_label


def div_data(data,div):
	X = data[:-div]
	Y = data[div: ]
	return X,Y


def stand(x):
	start = time.time()
	mean = x.mean()
	std = x.std()
	x = (x - mean) / std
	elapsed = (time.time() - start)
	print("stand time used:",round(elapsed,2))
	return x


def get_input(div = 1,lev = 1,block_num = 1,save_path = '/Data'):

	path = save_path +str(div)+'_'+str(lev)+'_'+str(block_num)+'.npy'
	my_file = Path(path)
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	if not my_file.exists():
		phs = get_block_from(tar = 'phs',lev = 0,block_num = block_num,save_path = save_path)
		pt = get_block_list_from(tar = 'pt',lev = lev,block_num = block_num,save_path = save_path)
		u = get_block_list_from(tar = 'u',lev = lev,block_num = block_num,save_path = save_path)
		# fix(lev = 28,save_path = save_path) #fix 179*360 to 180*360
		v = get_block_list_from(tar = 'v',lev = lev,block_num = block_num,save_path = save_path)
		input_x_merge = [list(phs)] + list(pt) + list(u) + list(v)
		merge = np.array(input_x_merge)
		del input_x_merge
		print(merge.shape)
		input_x = np.stack([x for x in merge],axis=1)
		input_x = input_x[:-1]
		global_shape = input_x.shape

		np.save(path,input_x)
	else:
		half_path = save_path +str(div)+'_'+str(lev)+'_'+str(block_num)+'_half.npy'
		# half_path = save_path +str(div)+'_'+str(lev)+'_'+str(block_num)+'.npy'

		if lev == 32:
			if comm.size == 1:
				input_x = np.load(half_path)
				global_shape = input_x.shape
			else:
				global_shape,input_x = parallel_read_array(half_path,axis = 0,comm = comm)
			
		else:
			input_x = np.load(path)
			global_shape = input_x.shape
	return global_shape,input_x

def get_label(tar = 'phs',div = 1,lev = 1,block_num = 1,save_path = '/Data'):

	data = get_block_from(tar = tar,lev = 0,block_num = block_num,save_path = save_path)

	X,Y = div_data(data,div = div)
		
	return X,Y

def get_block_from(tar = 'phs',lev = 1,block_num = 1,save_path = '/Data'):

		# for l in range(lev):
			# for ite in range(1,block_num+1):
		comm = MPI.COMM_WORLD
		x = (block_num+1) * 1440
		l = lev
		path = save_path + tar+'_'+str(l)+'_'+str(x)+'.npy'
		if comm.size == 1:
			d = np.load(path)
		else:
			global_shape,d = parallel_read_array(path,axis = 0,comm = comm)

		# d = np.load(half_path)

		print(d.shape)
		return d

def get_block_list_from(tar = 'phs',lev = 1,block_num = 1,save_path = '/Data'):

		d = []
		for l in range(lev):
			# for ite in range(1,block_num+1):
			x = (block_num+1) * 1440
			path = save_path + tar+'_'+str(l)+'_'+str(x)+'.npy'
			d.append(np.load(path))
			# print(tar,l,d[l].shape)

		d = np.array(d)
		print(tar,d.shape)

		return d

def fix(lev = 0,save_path = 'Data/'):

	# if num in fixed: return
	it = 0
	if True:
		it += 1
		x = it * 1440
		my_file = Path(save_path + 'v'+'_'+str(lev)+'_'+str(x)+'.npy')
		while my_file.exists():

			v = np.load(save_path + 'v'+'_'+str(lev)+'_'+str(x)+'.npy')
			print('fix',v.shape)
			if v.shape[1] == 179:
				v= v.tolist()
				for m in range(len(v)):
					v[m].append([0]*360)
				v = np.array(v)
				
				np.save(save_path + 'v'+'_'+str(lev)+'_'+str(x)+'.npy',v)
			it += 1
			x = it * 1440
			my_file = Path(save_path + 'v'+'_'+str(lev)+'_'+str(x)+'.npy')

