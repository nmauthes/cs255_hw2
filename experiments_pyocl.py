import random
import time

import matplotlib.pyplot as plt

from PyOpenCLInner import pyocl_inner_product


def random_vector(length=10, val_range=(-10, 10)):
    return [random.randint(*val_range) for _ in range(length)]


if __name__ == '__main__':
    lengths = [n for n in range(10, 10000, 100)]
    times = []

    for n in lengths:
        x = random_vector(n)
        y = random_vector(n)

        start = time.time()
        pyocl_inner_product(x, y)
        run_time = time.time() - start

        times.append(run_time)

        print(str(n) + ': ', run_time)

    plt.title('Performance of inner product using threading')
    plt.xlabel('Vector length (n)')
    plt.ylabel('Runtime (sec)')
    plt.plot(lengths, times)
    plt.show()