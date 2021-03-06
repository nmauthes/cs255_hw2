'''
Computes the inner product of two vectors using threading.

:authors Kunal, Nick, Shashank
'''


import argparse
import os
import threading


def inner_product_parallel(x, y):
    '''
    Compute the inner products of two vectors using parallel threads.

    :param x: The first vector
    :param y: The second vector
    :return: Inner product of x and y
    '''

    xy_products = []

    threads = []
    for x_i, y_i in zip(x, y):
        t = threading.Thread(target=lambda a, b: xy_products.append(a * b), args=(x_i, y_i))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    return sum(xy_products)


def inner_product(x, y):
    '''
        A simple sequential function to compute the inner products of two vectors.

        :param x: The first vector
        :param y: The second vector
        :return: Inner product of x and y
    '''

    return sum([x_i * y_i for x_i, y_i in zip(x, y)])


def parse_vector_file(filename):
    '''
    A function for parsing vectors from text data in .txt file.

    :param filename: The path of the vector file
    :return: The parsed vectors x and y
    '''

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


parser = argparse.ArgumentParser(
    description='Computes the inner product of two vectors using threading.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions see the README'
)

parser.add_argument(
    'filename_with_vector_data',
    help='Path to the file containing comma-separated vector data'
)


if __name__ == '__main__':
    args = parser.parse_args()

    x, y = parse_vector_file(args.filename_with_vector_data)
    print(inner_product_parallel(x, y))