import os
import numpy as np
from scipy.stats import dirichlet
from scipy import special
import prob_utils as pu


# Implementation of " Bayesian estimation of Dirichlet mixture model with variational inference", Zhanyu Ma


N_data = 1000                       # num of points
K_data = 3                          # num of dirichlet distribution dimensions
I_data = 2                          # num of mixture dirichlet distribution
Pi_data = [0.35, 0.65]              # prior of mixture probabilities
U_data = [                          # actual dirichlet parameters
    [4.0, 12.0, 3.0],
    [10.0, 6.0, 2.0],
]

# Generate Sample Data

X_data_cluster_1 = np.random.dirichlet(U_data[0], size=int(Pi_data[0] * N_data)).astype(dtype=np.double)
X_data_cluster_2 = np.random.dirichlet(U_data[1], size=int(Pi_data[1] * N_data)).astype(dtype=np.double)
X_data = np.vstack([X_data_cluster_1, X_data_cluster_2])
np.random.shuffle(X_data)


# Extended Variational Inference
K_model = K_data
I_model = 4


"""
    this is my model parameter initialization
"""
Omega_model_initial = np.random.random(size=(K_model, I_model)).astype(np.double) * 10 + 6
Alpha_model_initial = np.random.random(size=(K_model, I_model)).astype(np.double) * 5 + 1
C_model_initial = np.random.randint(2, 9, size=(I_model, )).astype(dtype=np.double)

"""
    this is initialization said in paper, but it's not working at all as I have tried.
"""
# Omega_model_initial = np.ones(shape=(K_model, I_model), dtype=np.double) * 1.0
# Alpha_model_initial = np.ones(shape=(K_model, I_model), dtype=np.double) * 0.001
# C_model_initial = np.ones(shape=(I_model,), dtype=np.double) * 0.0001


Omega_model = Omega_model_initial.copy()
Alpha_model = Alpha_model_initial.copy()
C_model = C_model_initial.copy()

print("\n"*10)
print("Initial Model Parameters")
print("\tfor U ~ Gamma(Omega, Alpha)")
print("\t\tOmega : ", Omega_model)
print("\t\tAlpha : ", Alpha_model)



def Exp_ln_X(X, O, A, C, nb_run=100):
    result = []
    remain_mixture = np.where(C > 0.00001)[0]
    O = O[:, remain_mixture]
    A = A[:, remain_mixture]
    C = C[remain_mixture]

    if len(remain_mixture) > 1:

        for i in range(nb_run):
            u = np.random.gamma(shape=O, scale=1.0/A).transpose().astype(np.double)
            pi = np.random.dirichlet(C, size=1)[0].astype(np.double)
            result.append(pu.logf_X_given_U_PI(X, u, pi))

        print("mean : ", np.mean(result))
        print("target : ", pu.logf_X_given_U_PI(X, U_data, Pi_data))
        print("")
    else:
        index = remain_mixture[0]


for step in range(1000):

    # Get a Fixed Point of U
    U_Fixpoint = Omega_model / Alpha_model  #(K, I)

    # Calculate Expectation of Z

    # equation 29
    P = np.zeros(shape=(I_model, ), dtype=np.double)
    for i in range(int(I_model)):
        p = [
            (
                special.digamma(U_Fixpoint[:, i].sum())
                - special.digamma(U_Fixpoint[k, i])
            )
            * U_Fixpoint[k, i]
            * (
                special.digamma(Omega_model[k, i]) - np.log(Alpha_model[k, i])  # Expectation of ln(Uki)
                - np.log(U_Fixpoint[k, i])
            )
            for k in range(K_model)
        ]
        P[i] = np.sum(p)

    # equation 30
    Rho = np.zeros(shape=(N_data, I_model), dtype=np.double)
    for i in range(I_model):
        # factor 1 is missing in equation 30 in original paper, may be the paper is wrong
        factor1 = special.loggamma(U_Fixpoint[:, i].sum()) - special.loggamma(U_Fixpoint[:, i]).sum()
        factor2 = special.digamma(C_model[i]) - special.digamma(C_model.sum())
        for n in range(N_data):
            Rho[n, i] = np.exp(
                    factor1
                    + factor2
                    + P[i] +
                    (
                            (U_Fixpoint[:, i] - 1.0)
                            * np.log(X_data[n, :])
                    ).sum()
            )

    Exp_Z = Rho / Rho.sum(axis=1, keepdims=True)
    Update_model = np.zeros(Alpha_model.shape)


    # algorithm 1
    for i in range(I_model):
        for k in range(K_model):
            Alpha_model[k, i] = Alpha_model_initial[k, i] - (Exp_Z[:, i] * np.log(X_data[:, k])).sum()
            Omega_model[k, i] = Omega_model_initial[k, i] + Exp_Z[:, i].sum() * U_Fixpoint[k, i] * (special.digamma(U_Fixpoint[:, i].sum()) - special.digamma(U_Fixpoint[k, i]))
        
        # the parameter C is likely to explode and causing the loop not convergent.
        # so I multiple it with 0.1
        C_model[i] = C_model[i] * 0.1 + Exp_Z[:, i].sum()


    Exp_Pi =  C_model / C_model.sum()
    
    arg_sort_index = np.argsort(Exp_Pi)[::-1]
    Exp_Pi = Exp_Pi[arg_sort_index]
    Exp_U = (Omega_model / Alpha_model).transpose()[arg_sort_index, :]
 

    if step % 50 == 0:
        print("End of step ", step+1)
        print("Omega : ", Omega_model.transpose())
        print("Alpha : ", Alpha_model.transpose())
        print("C : ", C_model)
        print("Exp_Pi : ", ','.join(['%0.4f'%d for d in Exp_Pi]))
        print("Exp_U : ", Exp_U)
        print("")

    if step % 50 == 0:
        Exp_ln_X(X_data, Omega_model, Alpha_model, C_model)


print("End of step ", step+1)
print("Omega : ", Omega_model.transpose())
print("Alpha : ", Alpha_model.transpose())
print("C : ", C_model)
print("Exp_Pi : ", ','.join(['%0.4f'%d for d in Exp_Pi]))
print("Exp_U : ", (Omega_model / Alpha_model).transpose())
print("")

print("U_data : ", U_data)
print("Pi_data : ", Pi_data)
