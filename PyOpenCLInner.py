'''
Computes the inner product of two vectors using PyOpenCL.

:authors Kunal, Nick, Shashank
'''

import argparse
#import pyopencl as cl


# Code goes here


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
    args = parser.parse_args()# Main method goes here