import numpy as np

def writeParams(W_k, b_k, omega, my, ypsilon, filename):
    try:
        file = open(filename, 'x')
    except FileExistsError:
        file = open(filename, 'w')

    Wk_str = ""
    for W in W_k:
        for w in W:
            for i in w:
                Wk_str += str(i) + ','
    Wk_str += '\n'

    bk_str = ""
    for B in b_k:
        for b in B:
            bk_str += str(b[0]) + ','
    bk_str += '\n'

    omega_str = ""
    for o in omega:
        omega_str += str(o[0]) + ','
    omega_str += '\n'

    my_str = str(my[0][0]) + '\n'
    
    ypsilon_str =""
    for y in ypsilon:
        ypsilon_str += str(y[0]) + ','
    ypsilon_str += '\n'

    file.write(Wk_str + bk_str + omega_str + my_str+ypsilon_str)
    file.close()


def readParams(K, d, I, filename):
    try:
        file = open(filename, 'r')
    except FileExistsError:
        print("Kunne ikke finne", filename)

    w_k = np.zeros((K, d, d))
    b_k = np.zeros((K, d))
    omega = np.zeros(d)
    my = np.zeros(1)
    ypsilon = np.zeros((I,1))
    

    W = file.readline().split(',')
    B = file.readline().split(',')
    O = file.readline().split(',')
    M = file.readline()
    Y = file.readline().split(',')

    file.close()

    for k in range(K):
        for w in range(d):
            for i in range(d):
                w_k[k][w][i] = float(W[0])
                W.pop(0)

    for k in range(K):
        for i in range(d):
            b_k[k][i] = float(B[0])
            B.pop(0)
    b_k.resize(K,d,1)

    for i in range(d):
        omega[i] = float(O[i])
    omega.resize(d,1)
    
    my[0] = float(M)
    my.resize(1,1)
    
    for y in range(I):
        ypsilon[y][0] = float(Y[y])

    return w_k, b_k, omega, my, ypsilon
