import matplotlib.pyplot as plt
import numpy as np
from math import exp

def logistic(theta, x):
    return 1 / (1 + exp(-theta * x))

if __name__=='__main__':
    x = np.arange(-10,10,0.2)
    p1, = plt.plot(x, np.asarray(list(map(lambda x:logistic(1,x), x))), 'r-', label='1')
    p2, = plt.plot(x, np.asarray(list(map(lambda x:logistic(7,x), x))), 'y-', label='7')
    p3, = plt.plot(x, np.asarray(list(map(lambda x:logistic(-3,x), x))), 'g-', label='-3')
    p4, = plt.plot(x, np.asarray(list(map(lambda x:logistic(-1,x), x))), 'o-', label='-1')
    plt.legend(loc='best')
    legend = plt.legend(handles=[p1,p2,p3,p4], title="Valori ale parametrului theta", loc='best', fontsize='small', fancybox=True)
    plt.show()