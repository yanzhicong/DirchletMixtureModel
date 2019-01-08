import os
import numpy as np
import prob_utils as pu
from DMM import DirichletMixtureModel


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


# Extended Variational Inference
model = DirichletMixtureModel(K=K_data, I=4)
model.extended_vi(X_data, validate_callback=lambda O,A,C:Exp_ln_X(X_data, O, A, C))

print("U_data : ", U_data)
print("Pi_data : ", Pi_data)


