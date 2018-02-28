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
        }
        """).build()

    #kernel = prg.inner_product
    #kernel.set_scalar_arg_dtypes( [None, np.int32, None] )
    prg.inner_product(queue, a.shape, None, a_dev.data, b_dev.data, result.data)

    #result = np.zeros(a.shape, np.int32)
    #future = cl.enqueue_copy(queue, result, result_prg)
    #future.wait()

    print(sum(result))


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
