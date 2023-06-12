import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from Classes import *

##Auxiliary functions


def trans(S, mg):
    if np.isscalar(mg):
        return (S.mu - S.r)/((S.sigma**2)*mg)
    if isinstance(mg, np.ndarray):
        if mg.ndim == 1:
            return np.flip((S.mu - S.r)/((S.sigma**2)*mg))
        else:
            return np.fliplr((S.mu - S.r)/((S.sigma**2)*mg))
        
def Usp(S , c):
    """returns utility social planner for given eta and certainty equivalent c""" 
    eta = S.eta
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1-eta)-1)/(1-eta)
    
def invUsp(S, u):
    eta = S.eta
    if eta == 1:
        return np.exp(u)
    else:
        return ((1-eta)*u + 1) ** (1 / (1- eta))

def label(x):
    if x == objG:
        return 'in terms of G'
    if x == objM:
        return 'in terms of M'
    if x == Adam:
        return 'Adam'
    if x == GD:
        return 'GD'
    if x == goal_function:
        return 'goal function'
def lin(x):
    return x
def log(x):
    return np.log(x)
def quad(x):
    return x**2
def exp(x):
    return np.exp(x)

def costlabel(x):
    if x == lin:
        return 'linear cost'
    if x == log:
        return 'logarithmic cost'
    if x == quad:
        return 'quadratic cost'
    if x == exp:
        return 'exponential cost'

def findIndex(sorted_array,l,u):
    left_idx = np.searchsorted(sorted_array, l, side='left')
    right_idx = np.searchsorted(sorted_array, u, side='right')
    indices = np.arange(left_idx, right_idx)
    return indices

def goal_function(S, g):
    eta = S.eta
    Gamma = S.Gamma
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T

    sum = 0


    for i in range(1, len(g)):
        mi = optimalDecision(S, g[i-1],g[i])
        def vce(g):
            return Usp(S, np.exp(r*T + (mu - r)*mi*T-.5*g*(sigma**2)*(mi**2)*T)) 
        if isinstance(Gamma, ECDF):
            indices = np.where((Gamma.x >= g[i-1]) & (Gamma.x <= g[i]))[0]
            E = np.sum(vce(Gamma.x[indices])) / len(Gamma.x)
            sum +=E          
        else:     
            integrand = lambda g: vce(g)*Gamma.pdf(g)
            E,_ = integrate.quad(integrand, g[i-1], g[i])
            sum += E

    return sum



#Algorithm section--
def optimalDecision(S, l=None, u=None):
    if l is None:
        l = S.a
    if u is None:
        u = S.b
    eta = S.eta
    Gamma = S.Gamma
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T

    def root(m):
        def h(g):
            return np.exp(g*.5*(sigma**2)*(eta-1)*T*(m**2))
        if isinstance(Gamma, ECDF):
            indices = findIndex(Gamma.x, l, u)
            samples = Gamma.x[indices]
            E1 = np.sum(samples * h(samples)) #take all even indices
            E2 = np.sum(h(samples)) #take all odd indices 
            return m-(mu-r)/((sigma**2)*E1/E2)        
        
        else:
            if S.MC:
                indices = findIndex(S.sample, l, u)
                samples = S.sample[indices]
                E1 = np.sum(samples *h(samples))/len(samples) 
                E2 = np.sum(h(samples))/len(samples)

            else:
                integrand1 = lambda g: g * h(g) * Gamma.pdf(g)
                E1,_ = integrate.quad(integrand1, l, u)
                integrand2 = lambda g: h(g) * Gamma.pdf(g)
                E2,_ = integrate.quad(integrand2, l, u)
                x = 5
            return m-(mu-r)/((sigma**2)*E1/E2)

    bracket = trans(S, np.array([l,u]))
    x0 = .5 * np.sum(bracket)

    sol = optimize.root_scalar(root, method = 'brentq', bracket = bracket, x0 = x0)
    return sol.root

def objG(S, g):
    n = S.n
    H = lambda x,y: 2/((1/x)+(1/y))
    residuals = np.zeros(n-1)
    g1 = trans(S, optimalDecision(S, g[0],g[1]))
    for i in range(1,n):
        g2 = trans(S, optimalDecision(S, g[i],g[i+1]))
        residuals[i-1] = (g[i]-H(g1,g2))**2
        g1 = g2
    sum = np.sum(residuals)
    return np.sum(residuals)

def objM(S, m):
    n = S.n
    residuals = np.zeros(n-1)
    m1 = optimalDecision(S, trans(S, m[1]), trans(S, m[0]))
    for i in range(1, n):
        m2 = optimalDecision(S, trans(S, m[i+1]), trans(S, m[i]))
        residuals[i-1] = (m[i] - 0.5 * (m1 + m2))**2
        m1 = m2
    return np.sum(residuals)

def gradient(S, f, x):
    """

    """
    eps = 1e-6  # small value for epsilon
    n = x.shape[0]  # number of dimensions
    
    grad = np.zeros(n)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (f(S, x_eps) - f(S, x)) / eps    
    return grad

#Algorithms:

def GD(S, obj = objG, allpath =0, tolerance=1e-2,  max_iterations=1000, learning_rate=1):
    a = S.a
    b = S.b
    n = S.n
    
    # Set the initial guess for the partition
    if obj == objM:
        partition = np.linspace(trans(S,b), trans(S,a), n+1)
        learning_rate = 1 * learning_rate
        #partition = trans(S,np.linspace(a, b, n+1))
    else:
        #partition = trans(S, np.linspace(trans(S,b), trans(S,a), n+1))
        partition = np.linspace(a, b, n+1)

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(S, obj, partition)
        maxgradnew = np.abs(grad).max() 

        d =1
        if maxgradnew > 2 * maxgradold:
            print("\n slow down\n")
            d = maxgradold/maxgradnew
        # Adjust the partition using the gradient
        partition[1:-1] -= learning_rate * grad[1:-1] * d

        # Enforce the constraints a=g0 and gn=b or m*(b) = m0 and m*(a) = mn
        if obj == objM:

            partition[0] = trans(S,b)
            partition[-1] =  trans(S,a) 
        else:
            partition[0] = a
            partition[-1] = b

        if i > 3:
            difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
            if difference <  1e-2 *tolerance:
                print('\n take the middle')
                partition = .5 * (np.array(partition)+np.array(path[-1]))

        path.append(partition.copy())

        #Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('unfeasible region')
            break

        # Check for convergence
        if maxgradnew < tolerance:
            break

        maxgradold = maxgradnew
    if allpath == 1:    
        return path
    return path[-1]

def Adam(S, obj = objG, allpath = 0, tolerance=1e-6, max_iterations=1000, learning_rate=.1,  epsilon=1e-8,  beta1=0.9, beta2=0.999):
    """
    Adam gradient descent algorithm for optimizing the objective function
    """
    a = S.a
    b = S.b
    n = S.n
    # Set the initial guess for the partition
    if obj == objM:
        partition = np.linspace(trans(S,b), trans(S,a), n+1)
        #partition = trans(S,np.linspace(a, b, n+1))
    else:
        #partition = trans(S, np.linspace(trans(S,b), trans(S,a), n+1))
        partition = np.linspace(a, b, n+1)

    # Initialize the moment estimates for the gradient and its squared magnitude
    m = np.zeros_like(partition)
    v = np.zeros_like(partition)

    path = []
    path.append(partition.copy()) # add the initial guess to the path
    
    maxgradold = 9999

    for i in tqdm(range(max_iterations)):
        # Compute the gradient of the objective function
        grad = gradient(S, obj, partition)
        maxgradnew = np.abs(grad).max()

        d =1
        if maxgradnew > 2 * maxgradold:
            print("   doop")
            d = maxgradold/maxgradnew
        
        # Update the moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias-correct the moment estimates
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))
        
        # Adjust the partition using the Adam update rule
        partition[1:-1] -= d* learning_rate * m_hat[1:-1] / (np.sqrt(v_hat[1:-1]) + epsilon)

        # Enforce the constraints a=g0 and gn=b or m*(b) = m0 and m*(a) = mn
        if obj == objG:
            partition[0] = a
            partition[-1] = b
        else:
            partition[0] = trans(S,b)
            partition[-1] =  trans(S,a) 

        if i > 3:
            difference = np.sum(np.abs(np.array(partition)-np.array(path[-2])))
            if difference <  tolerance:
                print('ploop')
                partition = .5 * (np.array(partition)+np.array(path[-1]))


        path.append(partition.copy())

        # Check whether partition is in ascending order
        if np.all(np.diff(path) > 0) == False:
            print('unfeasible region')
            break

        # Check for convergence
        if np.abs(grad).max() < tolerance:
            break

        maxgradold = maxgradnew

    if allpath == 1:    
        return path
    return path[-1]

