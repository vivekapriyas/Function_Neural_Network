import numpy as np

def writeParams(W_k, b_k, omega, my, ypsilon, J, itr, filename):
    try:
        file = open('parameters/'+filename, 'x')
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
    
    J_str = ""
    for j in J:
        J_str += str(j) + ','
    J_str += '\n'
    
    itr_str = str(itr) + '\n'

    file.write(Wk_str + bk_str + omega_str + my_str+ypsilon_str+J_str+itr_str)
    file.close()


def readParams(K, d, I, N, filename):
    try:
        file = open('parameters/'+filename, 'r')
    except FileExistsError:
        print("Kunne ikke finne", filename)

    w_k = np.zeros((K, d, d))
    b_k = np.zeros((K, d))
    omega = np.zeros(d)
    my = np.zeros(1)
    ypsilon = np.zeros((I,1))
    J = np.zeros(N) 
    

    W = file.readline().split(',')
    B = file.readline().split(',')
    O = file.readline().split(',')
    M = file.readline()
    Y = file.readline().split(',')
    Jl = file.readline().split(',')
    itr = file.readline()

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

    
    for j in range(N):
        J[j] = float(Jl[j])

    return w_k, b_k, omega, my, ypsilon, J, int(itr)

def writeScale(aV,bV, aT, bT, filename):
    try:
        file = open('parameters/'+filename, 'x')
    except FileExistsError:
        file = open(filename, 'w')
    
    file.write(str(aV)+'\n'+str(bV)+'\n'+str(aT)+'\n'+str(bT))
    file.close()

    
def readScale(filename):
    try:
        file = open('parameters/'+filename, 'r')
    except FileExistsError:
        print("could not find file", filename)
    
    aV = float(file.readline())
    bV = float(file.readline())
    aT = float(file.readline())
    bT = float(file.readline())
    
    return aV, bV, aT, bT