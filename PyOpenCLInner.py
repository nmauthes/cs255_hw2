'''
Computes the inner product of two vectors using PyOpenCL.

:authors Kunal, Nick, Shashank
'''
from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import os
import argparse

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = ':'

def parse_vector_file(filename):
    if not os.path.exists(filename):
        raise Exception('Vector file not found')

    x = []
    y = []

    with open(filename, 'r') as f:
        for line in f:
            x_i, y_i = line.split(',')

            x.append(int(x_i))
            y.append(int(y_i))
    
    return x, y

def pyocl_inner_product(x , y):
    '''Creates two np arrays with the provided data.
       A third np array is created that stores the result.
       The OpenCL kernel multiplies the two arrays in parallel.
       A barrier function ensures that all multiplication
       operations have finished before proceeding. 

       Each work item is responsible for adding its neighbour at
       some distance which is a multiple of 2 provided its 
       global id is divisible by a multiple of 2.

       Effectively the for loop terminates in O (log n) steps.
    '''
    a = np.array(x).astype(np.int32)
    b = np.array(y).astype(np.int32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    a_dev = cl_array.to_device(queue, a)
    b_dev = cl_array.to_device(queue, b)

    result = cl_array.empty_like(a_dev)

    prg = cl.Program(ctx, """
        __kernel void inner_product(__global const int *a,
        __global const int *b, __global int *c)
        {
            int gid = get_global_id(0);
            c[gid] = a[gid] * b[gid];
            barrier(CLK_GLOBAL_MEM_FENCE);

            int group_size = get_global_size(0);
            
            for(int i = 2; i/2<=group_size; i*=2)
            {
                if(gid % i == 0 && (gid + i/2) <=group_size)
                    c[gid] += c[gid + i/2];
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        """).build()

    prg.inner_product(queue, a.shape, None, a_dev.data, b_dev.data, result.data)

    print(result)


parser = argparse.ArgumentParser(
    description='Computes the inner product of two vectors using PyOpenCL.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions see the README'
)

parser.add_argument(
    'filename_with_vector_data',
    help='Path to the file containing comma-separated vector data'
)

if __name__ == '__main__':
    args = parser.parse_args()

    x, y =parse_vector_file(args.filename_with_vector_data)
    pyocl_inner_product(x, y)
