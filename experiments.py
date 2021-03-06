import random
import time

import numpy as np
import matplotlib.pyplot as plt

from ThreadInner import inner_product_parallel


def random_vector(length=10, val_range=(-10, 10)):
    return [random.randint(*val_range) for _ in range(length)]


if __name__ == '__main__':
    lengths = [n for n in range(10, 10000, 100)]
    times = []

    for n in lengths:
        x = random_vector(n)
        y = random_vector(n)

        start = time.time()
        inner_product_parallel(x, y)
        run_time = time.time() - start

        times.append(run_time)

        print(str(n) + ': ', run_time)

    fit = np.polyfit(lengths, times, deg=1)

    plt.title('Performance of inner product using threading')
    plt.xlabel('Vector length (n)')
    plt.ylabel('Runtime (sec)')
    plt.plot(lengths, times, color='red')
    plt.plot(lengths, fit[0] * np.asarray(lengths, np.float64) + fit[1], color='black', linestyle='dashed', alpha=0.5)
    plt.show()