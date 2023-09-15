import numpy as np
import mpmath
import time
from scipy.special import roots_legendre


def gauss_legendre(f, a, b, n):

    t, A = roots_legendre(n)

    x = [((b-a)/2*t_i + (b+a)/2) for t_i in t]

    fx = [f(x_i) for x_i in x]

    I = sum([fx_i * A_i for fx_i, A_i in zip(fx, A)]) * ((b-a)/2)

    return I


if __name__ == '__main__':
    def f(x): return 1/x

    a = 1
    b = 2
    n = 50

    R = np.log(2)

    start = time.time()
    for i in range(1000):
        I = mpmath.quad(f, [a, b], method='gauss-legendre')
    end = time.time()

    print('Método da biblioteca')
    print(f'Tempo de execução: {(end-start)/1000}')
    print(f'Erro: {((I-R)/R)*100}%')

    start = time.time()
    for i in range(1000):
        I = gauss_legendre(f, a, b, n)
    end = time.time()

    print('\nMétodo implementado')
    print(f'Tempo de execução: {(end-start)/1000}')
    print(f'Erro: {((I-R)/R)*100}%')
