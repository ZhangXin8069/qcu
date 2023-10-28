import cupy as cp
import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Set the number of GPUs and select the appropriate one for the current rank
    num_gpus = cp.cuda.runtime.getDeviceCount()
    current_gpu = rank % num_gpus
    cp.cuda.Device(current_gpu).use()

    # Define the size of the array
    array_size = 10000000  # Adjust this to your desired size

    if rank == 0:
        # Allocate CPU memory
        start_cpu = time.time()
        host_data = cp.arange(array_size, dtype=cp.float32)
        end_cpu = time.time()

        # Send data to other processes
        for i in range(1, comm.Get_size()):
            comm.send(host_data, dest=i)

        # Perform a CPU computation
        result_cpu = cp.sqrt(host_data)

        # Measure CPU time
        cpu_time = end_cpu - start_cpu
        print(f"CPU Time: {cpu_time} seconds")

    else:
        # Receive data from rank 0
        host_data = comm.recv(source=0)

        # Perform GPU computation
        start_gpu = time.time()
        gpu_data = cp.sqrt(host_data)
        end_gpu = time.time()

        # Measure GPU time
        gpu_time = end_gpu - start_gpu
        print(f"Rank {rank} - GPU Time: {gpu_time} seconds")

if __name__ == '__main__':
    for i in range(10):
        main()