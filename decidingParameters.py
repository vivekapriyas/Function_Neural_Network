import numpy as np
from project_2_data_acquisition import generate_data, concatenate
from files import writeParams, readParams
import matplotlib.pyplot as plt

#define lists of filenames where the trained parameters will be stored 
def filenameList(parameter_list, parameter):
    filenames_A = []
    filenames_P = []
    for i in range(len(parameter_list)):
        num = parameter_list[i]
        filenames_A.append('trainingParams_A{}{}'.format(parameter, num))
        filenames_P.append('trainingParams_P{}{}'.format(parameter, num))
    return filenames_A, filenames_P

def plotParams(parameter_list, parametername, K = None, d = None, batchsize = None, N=5000):
    fig, axs = plt.subplots(1, 2, sharey = True)
    fig.set_figheight(7)
    fig.set_figwidth(10)
    
    #plot adam
    ax = axs[0]
    for i in range(len(parameter_list)):
        parameter = parameter_list[i]
        if parametername == "batch":
            fig.suptitle('J vs. iterations for different batchsizes for stochastic gradient descent', fontsize=12)
            W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_Abatch, itr_A0 = readParams(K, d_0, parameter, N, filename = "trainingParams_Abatch{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Abatch/parameter, label = r"$batchsize ={}$".format(parameter))
            
            
        elif parametername == "K":
            fig.suptitle('J vs. iterations for different values of hidden layers K', fontsize=12)
            W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_AK, itr_A0 = readParams(parameter, d_0, batchsize, N, filename = "trainingParams_AK{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_AK/batchsize, label = r"$K ={}$".format(parameter))
        
        elif parametername == "tau":
            #fig.suptitle(r'J vs. iterations for different values of the learning parameter \tau', fontsize=12)
            W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_Atau, itr_A0 = readParams(K, d_0, batchsize, N, filename = "trainingParams_Atau{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Atau/batchsize, label = r"$\tau =%.4f$"%parameter)
            
        elif parametername == "d":
            fig.suptitle('J vs. iterations for different dimensions d', fontsize=12)
            W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_Ad, itr_A0 = readParams(K, parameter, batchsize, N, filename = "trainingParams_Ad{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Ad/batchsize, label = r"$d ={}$".format(parameter))
            
        elif parametername == "h":
            fig.suptitle('J vs. iterations for different values the stepsize h', fontsize=12)
            W_A0, b_A0, omega_A0, mu_A0, ypsilon_A0, J_Ah, itr_A0 = readParams(K, d_0, batchsize, N, filename = "trainingParams_Ah{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Ah/batchsize, label = r"$h ={}$".format(parameter))
        
        else:
            print("The parametername must be either \"batch\", \"K\", \"tau\", \"d\" or \"h\"")
        
    ax.set_title("Adam descent")
    ax.set_yscale("log")
    ax.set_ylabel("Loss function, J")
    ax.set_xlabel("Iterations, N")
                  
    ax.legend()
    
    
    #plot plain
    ax = axs[1]
    for i in range(len(parameter_list)):
        parameter = parameter_list[i]
        
        if parametername == "batch":
            W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_Pbatch, itr_P0 = readParams(K, d_0, parameter, N, filename = "trainingParams_Pbatch{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Pbatch/parameter, label = r"$batchsize ={}$".format(parameter))
        
        elif parametername == "K":
            W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_PK, itr_P0 = readParams(parameter, d_0, batchsize, N, filename = "trainingParams_PK{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_PK/batchsize, label = r"$K ={}$".format(parameter))
            
        elif parametername == "tau":
            W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_Ptau, itr_P0 = readParams(K, d_0, batchsize, N, filename = "trainingParams_Ptau{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Ptau/batchsize, label = r"$\tau =%.4f$"%parameter)
    
        elif parametername == "d":
            W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_Pd, itr_P0 = readParams(K, parameter, batchsize, N, filename = "trainingParams_Pd{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Pd/batchsize, label = r"$d ={}$".format(parameter))
            
        elif parametername == "h":
            W_P0, b_P0, omega_P0, mu_P0, ypsilon_P0, J_Ph1, itr_P0 = readParams(K, d_0, batchsize, N, filename = "trainingParams_Ph{}".format(parameter))
            ax.plot(np.linspace(0,N,N), J_Ph1/batchsize, label = r"$h ={}$".format(parameter))
            
        else:
            print("The parametername must be either \"batch\", \"K\", \"tau\", \"d\" or \"h\"")
              
    ax.set_title(r"Plain vanilla gradient descent")
    ax.set_yscale("log")
    ax.set_ylabel("Loss function, J")
    ax.set_xlabel("Iterations, N")
    ax.legend()

    plt.show()
 



batch0 = generate_data(0)

p0_tilde = batch0['P']

N = 5000
tol = 1E-10 
I_0 = p0_tilde.shape[1]
batchsize = I_0
d_0 = p0_tilde.shape[0]    
h = 0.1     
tau = 0.001  
K = 50


"""
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
    """