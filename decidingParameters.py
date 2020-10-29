import numpy as np
from project_2_data_acquisition import generate_data, concatenate
from files import writeParams, readParams

#define lists of filenames where the trained parameters will be stored 
def filenameList(parameter_list, parameter):
    filenames_A = []
    filenames_P = []
    for i in range(len(liste)):
        num = liste[i]
        filenames_A.append('trainingParams_A{}{}'.format(parameter, num))
        filenames_P.append('trainingParams_P{}{}'.format(parameter, num))
    return filenames_A, filenames_P


#Input data
batch0 = generate_data(0)

p0_tilde = batch0['P']
T0_tilde, aT0, bT0 = scale(batch0['T'])
T0_tilde.resize(1,T0_tilde.shape[0])
I_0 = p0_tilde.shape[1]

#General variables
N = 5000
tol = 1E-10
batchsize = I_0
d_0 = p0_tilde.shape[0]    
h = 0.1     
tau = 0.001  
K = 50


#test with diffrent batchsizes
batchsize_list = np.arange(I_0/4, I_0+1,I_0/4).astype(int) 

filenames_batchsize_A, filenames_batchsize_P = filnameList(batchsize_list, "batch")    
    
for i in range(len(batchsize_list)):
    batchsize = batchsize_list[i]
    #train for different batchsizes with adam descent 
    mu_A0, omega_A0, W_A0, b_A0, J_Abatch, ypsilon_A0, itr_A0 = trainingAlgorithm(K, d_0, h, tau, p0_tilde, T0_tilde, eta, sigma, eta_div, sigma_div, N, tol, batchsize, "adam")
    writeParams(W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_Abatch, itr_A0, filename = filenames_batchsize_A[i])
    
    #train for different batchsizes with plain vanilla gradient descent
    mu_P0, omega_P0, W_P0, b_P0, J_Pbatch, ypsilon_P0, itr_P0 = trainingAlgorithm(K, d_0, h, tau, p0_tilde, T0_tilde, eta, sigma, eta_div, sigma_div, N, tol, batchsize, "plain")
    writeParams(W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_Pbatch, itr_P0, filename = filenames_batchsize_P[i])