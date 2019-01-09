import os
import numpy as np
from scipy.stats import dirichlet
from scipy import special
import prob_utils as pu



class InfiniteInvertedDirichletMixtureModel(object):
    """
        
    """

    def __init__(self,  K=None, I=None, U=None, PI=None, T=100):
        """
        :param I:   number of mixture components
        :param K:   number of variable dimensions
        :param U:   the dirichlet distribution parameters, shape : [K, I]
        :param PI:  the component coefficient, shape : [I,]
        """

        # model define
        if K is None or I is None:
            assert U is not None
            self.K = U.shape[0]
            self.I = U.shape[1]
        else:
            self.K = K
            self.I = I

        # assume U ~ Gamma(Omega (shape param), 1.0/Alpha (scale param))
        # random initialize Omega and Alpha
        self.Omega = np.random.random(size=(K, I)).astype(np.double) * 10 + 6
        self.Alpha = np.random.random(size=(K, I)).astype(np.double) * 5 + 1

        # assume PI ~ Dir(C)
        # random initialize C
        self.C = np.random.randint(2, 9, size=(I,)).astype(dtype=np.double)

        # this is initialization said in paper, but it's not working at all as I have tried.
        # self.Omega = np.ones(shape=(K, I), dtype=np.double) * 1.0
        # self.Alpha = np.ones(shape=(K, I), dtype=np.double) * 0.001
        # self.C = np.ones(shape=(I,), dtype=np.double) * 0.0001

        if U is not None:
            self.U = U
        else:
            self.U = np.random.gamma(shape=self.Omega, scale=1.0/self.Alpha)

        if PI is not None:
            self.PI = PI
        else:
            self.PI = np.random.dirichlet(alpha=self.C)


    def log_x(self, x):
        pass

    def extended_vi(self, X, max_steps=500, callback_interval=50, validate_callback=None):
        """
            Use extended variation inference method to solve the posterior distribution of 
            DMM parameters (U, PI) given the observed data X

            Implementation of " Bayesian estimation of Dirichlet mixture model with variational inference", 
            Zhanyu Ma

        """
        N = X.shape[0]
        NewOmega = self.Omega.copy()
        NewAlpha = self.Alpha.copy()
        NewC = self.C.copy()

        for step in range(max_steps):

            # Get a Fixed Point of U
            U_Fixpoint = NewOmega / NewAlpha  #(K, I)

            # Calculate Expectation of Z

            # equation 29
            P = np.zeros(shape=(self.I, ), dtype=np.double)
            for i in range(int(self.I)):
                p = [
                    (
                        special.digamma(U_Fixpoint[:, i].sum())
                        - special.digamma(U_Fixpoint[k, i])
                    )
                    * U_Fixpoint[k, i]
                    * (
                        special.digamma(NewOmega[k, i]) - np.log(NewAlpha[k, i])  # Expectation of ln(Uki)
                        - np.log(U_Fixpoint[k, i])
                    )
                    for k in range(self.K)
                ]
                P[i] = np.sum(p)

            # equation 30
            Rho = np.zeros(shape=(N, self.I), dtype=np.double)
            for i in range(self.I):
                # factor 1 is missing in equation 30 in original paper, may be it's a mistake
                factor1 = special.loggamma(U_Fixpoint[:, i].sum()) - special.loggamma(U_Fixpoint[:, i]).sum()
                factor2 = special.digamma(NewC[i]) - special.digamma(NewC.sum())
                for n in range(N):
                    Rho[n, i] = np.exp(
                            factor1
                            + factor2
                            + P[i] +
                            (
                                    (U_Fixpoint[:, i] - 1.0)
                                    * np.log(X[n, :])
                            ).sum()
                    )

            Exp_Z = Rho / Rho.sum(axis=1, keepdims=True)
            Update_model = np.zeros(NewAlpha.shape)

            # algorithm 1
            for i in range(self.I):
                for k in range(self.K):
                    NewAlpha[k, i] = self.Alpha[k, i] - (Exp_Z[:, i] * np.log(X[:, k])).sum()
                    NewOmega[k, i] = self.Omega[k, i] + Exp_Z[:, i].sum() * U_Fixpoint[k, i] * (special.digamma(U_Fixpoint[:, i].sum()) - special.digamma(U_Fixpoint[k, i]))
                
                # the parameter C is likely to explode and causing the loop not convergent.
                # so I multiple it with 0.1
                # NewC[i] = NewC[i] * 0.1 + Exp_Z[:, i].sum()

                NewC[i] = self.C[i] + Exp_Z[:, i].sum()

            Exp_Pi =  NewC / NewC.sum()
            
            arg_sort_index = np.argsort(Exp_Pi)[::-1]
            Exp_Pi = Exp_Pi[arg_sort_index]
            Exp_U = (NewOmega / NewAlpha).transpose()[arg_sort_index, :]


            if step % 50 == 0:
                print("End of step ", step+1)
                print("Omega : ", NewOmega.transpose())
                print("Alpha : ", NewAlpha.transpose())
                print("C : ", NewC)
                print("Exp_Pi : ", ','.join(['%0.4f'%d for d in Exp_Pi]))
                print("Exp_U : ", Exp_U)
                print("")

            if step % callback_interval == 0:
                validate_callback(NewOmega, NewAlpha, NewC)

        self.Omega = NewOmega
        self.Alpha = NewAlpha
        self.C = NewC


    @property
    def distribution_of_U(self):
        return self.Omega, self.Alpha

    @property
    def distribution_of_PI(self):
        return self.C

