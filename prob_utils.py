import numpy as np
from scipy.stats import dirichlet
from scipy.stats import gamma
from scipy.stats import multinomial


def f_x_given_u(x, u):
    return dirichlet.pdf(x, u)

def logf_x_given_u(x, u):
    return dirichlet.logpdf(x, u)

# def logf_x_given_


def f_X_given_u(X, u):
    return np.prod([dirichlet.pdf(x, u) for x in X])


def logf_X_given_u(X, u):
    return np.sum([dirichlet.logpdf(x, u) for x in X])




# def logf_X_given_U_PI(X, U, PI):
#
#
#     return np.sum([
#         dirichlet.pdf(x, ui) * pi
#         f_X_given_U(X, ui) * pi
#         for
#     ])


def f_X_given_Z_U(X, Z, U):

    '''
        X : [n-samples, m-dims]
        Z : [n-samples, i-mixtures]
        U : [i-mixtures, m-dims]
    '''
    assert X.shape[0] == Z.shape[0]
    assert X.shape[1] == U.shape[1]
    assert Z.shape[1] == U.shape[0]

    nN = X.shape[0]
    nI = X.shape[1]

    results = []
    for x, z in zip(X, Z):
        for i in range(nI):
            results.append(np.power(dirichlet.logpdf(x, U[i, :]), z[i]))
    return np.prod(results)


def logf_X_given_Z_U(X, Z, U):

    '''
        X : [n-samples, m-dims]
        Z : [n-samples, i-mixtures]
        U : [i-mixtures, m-dims]
    '''
    assert X.shape[0] == Z.shape[0]
    assert X.shape[1] == U.shape[1]
    assert Z.shape[1] == U.shape[0]

    nN = X.shape[0]
    nI = X.shape[1]

    results = []
    for x, z in zip(X, Z):
        for i in range(nI):
            results.append(z[i] * dirichlet.logpdf(x, U[i, :]))
    return np.sum(results)


def f_Z_given_PI(Z, PI):
    '''
        Z : [n-samples, i-mixtures]
        PI : [n-samples, i-mixtures]
    '''
    assert Z.shape[0] == PI.shape[0]
    assert Z.shape[1] == PI.shape[1]
    return np.prod(np.power(PI, Z))


def logf_Z_given_PI(Z, PI):
    '''
        Z : [n-samples, i-mixtures]
        PI : [n-samples, i-mixtures]
    '''

    assert Z.shape[0] == PI.shape[0]
    assert Z.shape[1] == PI.shape[1]

    return np.sum(Z * np.log(PI))


def f_PI(PI, c0):
    '''
        PI : [n-samples, i-mixtures]
        c0 : [i-mixtures]
    '''
    return np.prod([dirichlet.pdf(pi, c0) for pi in PI])


def logf_PI(PI, c0):
    '''
        PI : [n-samples, i-mixtures]
        c0 : [i-mixtures]
    '''
    return np.sum([dirichlet.logpdf(pi, c0) for pi in PI])



def f_U(U, Omega, Alpha):
    '''
        U : [i-mixtures, m-dims]
    '''
    nI, nM = U.shape

    if isinstance(Alpha, float):
        Alpha = np.ones((nI, nM)) * Alpha

    if isinstance(Omega, float):
        Omega = np.ones((nI, nM)) * Omega

    results = []

    for i in range(nI):
        for m in range(nM):
            results.append(gamma.pdf(U[i,m], Omega[i,m], scale=(1.0/Alpha[i,m])))
    return np.prod(results)


def logf_U(U, Omega, Alpha):
    '''
        U : [i-mixtures, m-dims]
        Omega : [i-mixtures, m-dims]
        Alpha : [i-mixtures, m-dims]
    '''
    nI, nM = U.shape

    if isinstance(Alpha, float):
        Alpha = np.ones((nI, nM)) * Alpha

    if isinstance(Omega, float):
        Omega = np.ones((nI, nM)) * Omega

    results = []

    for i in range(nI):
        for m in range(nM):
            results.append(gamma.logpdf(U[i,m], Omega[i,m], scale=(1.0/Alpha[i,m])))
    return np.sum(results)


def logf_X_U_PI_Z(X, U, PI, Z, c0, Omega, Alpha):
    
    _logf_X_given_Z_U = logf_X_given_Z_U(X, Z, U)
    _logf_Z_given_PI = logf_Z_given_PI(Z, PI)
    _logf_PI = logf_PI(PI, c0)
    _logf_U = logf_U(U, Omega, Alpha)

    return _logf_X_given_Z_U + logf_Z_given_PI + logf_PI + _logf_U



def logf_X_given_U_PI(X, U, PI):
    return np.sum([
        np.log(np.sum(
            [
                f_x_given_u(x, ui) * pii
                for ui, pii in zip(U, PI)
            ]
        ))
        for x in X
    ])
