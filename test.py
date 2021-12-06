import mpi4py.MPI as MPI


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

print(comm_rank,comm_size)


begin = comm_rank*5
end = begin + 5

print(comm_rank,begin,end)