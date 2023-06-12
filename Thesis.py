# This file is dedicated to produce all results needed in my thesis. Clarity is a must!
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from Classes import *
from Settingfunctions import *


def returndistributions(S):
    sigma = S.sigma
    mu = S.mu
    r = S.r
    T = S.T
    m_values = np.linspace(0.1, .5, 3)

    # Generate a standard normal distribution of outcomes
    n= 1000000
    Z = np.random.normal(size=n)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    def R(m, Z):
        return np.exp(r*T + (mu - r)*m*T-.5*(sigma**2)*(m**2)*T + m*sigma*Z*np.sqrt(T))

    # Loop over the m values and plot the distribution of R for each value
    # plot histograms for each m value
    for i, m in enumerate(m_values):
        R_values = [R(m, Z[j]) for j in range(n)]
        plt.hist(R_values, bins=500, alpha=0.5, label=f"m={m}", density=True)
        plt.axvline(np.mean(R_values), color=f'C{i}', linestyle='dashed', linewidth=5)


    # Add labels and legend to the plot
    ax.set_xlabel("R")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of R for different values of m")
    ax.legend()

    # Show the plot
    plt.show()
#returndistributions()

def MCcomparison(S, l=None, u=None):
    if l is None:
        l = S.a
    if u is None:
        u = S.b
       
    m = optimalDecision(S, l, u)
    plt.axvline(m, linewidth=1)
    S.Gamma=Gammas.unif()
    J = 10000
    samplesizes = [1e1, 1e3, 1e5]
    for ss in samplesizes:
        print(f'{ss}')
        M = np.zeros(J)
        for i in range(J):
            s1.sample = np.sort(s1.getSample(int(ss)))
            m = optimalDecision(S, l, u)
            M[i] = m
        plt.hist(M, bins=100, label=f'Samplesize = {ss}', alpha= .5)
    plt.legend()
    plt.xlabel('Solutions m s.t. r_2N(m) = 0')
    plt.show()

#MCcomparison(s1)         


def IntrinsicComparison(S, algorithm = GD, obj = objG, bench = False, P = None):
    '''
    n: is the amount of decisions. choose n >= 3 for best graphs

    adam: 1 if use Adams algorithm, 0 if use GD algorithm
    '''
    a = S.a
    b = S.b
    n = S.n
    for i, var in enumerate([1e-6]):
        print(f'Algorithm: {label(algorithm)}, Objective: {label(obj)}, Learning Rate={var}')
        if P.any():
            P= algorithm(S,obj=obj,allpath=1, max_iterations=1000, tolerance=var)

        print(f'Objective value: {obj(S,P[-1])} and goal value percentage difference :{(goal_function(S, P[-1]) -goal_function(S, benchmark))/goal_function(S, benchmark)}')
        for i in range(1,n-1):
            p_values = [(p[i],p[i+1]) for p in P]
            p1, p2 = zip(*p_values)
            plt.plot(p1, p2, '-o', label=f"{label(algorithm)} pair ({i},{i+1}): n={n}, Tolerance={var}")
            last_values = [(P[-1][i],P[-1][i+1])]
            p1, p2 = zip(*last_values)
            plt.scatter(p1, p2, s = 100, color = 'red')
    if bench:
        for i in range(1,n-1):
            p_values = [(benchmark[i],benchmark[i+1])]
            p1, p2 = zip(*p_values)
            plt.plot(p1, p2, '-o', label=f"{label(algorithm)} pair ({i},{i+1}): n={n}, Benchmark")
    if obj == objM:
        diagonal = [(x,x) for x in np.linspace(trans(S,b),trans(S,a),2)]
    else:
        diagonal = [(x,x) for x in np.linspace(a,b,2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")
    #plt.legend(fontsize=20)
    plt.show()
#IntrinsicComparison(s1, GD, objG, bench=False)


def ExtrinsicComparison(S, variants, field = objG):
    """
    S: the setting class
    variants: a list of lists containing [Adam/Gd, objM/objG]
    field: whether we plot the steps in the risk or decision fields of all variants
    field = 'objG': plot in terms of risk partitions
    field = 'objM':  plot in terms of decision partitions
    """
    a = S.a
    b = S.b
    n = S.n
    for variant in variants:
        print(f'Algorithm: {label(variant[0])}, Objective: {label(variant[1])}')
        P= variant[0](S, variant[1], allpath = 1, tolerance = 1e-3)        
        if variant[1]!=field: #translate if obj and field differ
            if variant[1] == objM:
                P = trans(S, np.array(P))
            if variant[1] == objG:
                P = trans(S, np.array(P))
        for i in range(1,n-1):
            p_values = [(p[i],p[i+1]) for p in P]
            p1, p2 = zip(*p_values)
            plt.plot(p1, p2, '-o', label=f" ({i},{i+1}):Algorithm: {label(variant[0])}, Objective: {label(variant[1])}")
    if field == objG:
        diagonal = [(x,x) for x in np.linspace(a,b,2)]
    if field == objM:
        diagonal = [(x,x) for x in np.linspace(trans(S,b),trans(S,a),2)]
    x1,x2 = zip(*diagonal)
    plt.plot(x1,x2,'-o', label="Feasible Boundary")
    plt.legend(fontsize=20)
    plt.show()
#Allvariants = [[Adam,objG],[GD,objG],[Adam, objM], [GD, objM]]
#ExtrinsicComparison(s1, Allvariants, objM)

GDvariants = [[GD, objM], [GD, objG]]
GDAdam = [[GD, objG], [Adam, objG]]
Single = [[GD, objG]]
#ExtrinsicComparison(s1, GDAdam, objM)




def consistency(S, RGamma, algorithm = GD,  obj = objG):
    S.Gamma = RGamma

    #P = algorithm(S, obj, tolerance=1e-3)
    P = benchmark
    # Plotting vertical lines at each point in P
    for point in P:
        plt.axvline(point, linewidth=1)

    samplesizes = [1e2, 1e3, 1e4]
    amount = 100
    for ss in samplesizes:
        print(f'samplesize = {ss}')
        Paths = []
        samples = RGamma.rvs((amount,int(ss)))
        for i in range(amount):
            print(f'ss={ss}, i={i}')
            EGamma =  ECDF(samples[i])
            S.Gamma =  EGamma
            P = algorithm(S, obj)[1:-1]
            Paths.append(P)
        plt.hist(np.ravel(np.array(Paths)), bins=100, alpha=0.5, label=f"samplesize =  {ss}")
    plt.legend()
    plt.show()

#consistency(s1, Gammas.unif(), GD, objG)



def progression(S, RGamma, algorithm = GD, obj = objG):
    """
    shows how one emperical estimation approaches a real partition over its iterations
    """
    S.Gamma = RGamma

    #Pgoal = algorithm(S,obj)
    Pgoal = benchmark
    N = 10000
    S.Gamma = Gammas.unif()
    P = algorithm(S, obj, allpath=1)
    print(f'Objective value: {obj(S,P[-1])} and goal value percentage difference :{(goal_function(S, benchmark)-goal_function(S, P[-1]))/goal_function(S, P[-1])}')
    chosenPercentages = [0, .2, .4, .6, .8]
    chosenSteps = [round(i * len(P)) for i in chosenPercentages] #note that round(i* len(P)) wil raise error
    chosenPartitions = [P[i] for i in chosenSteps]
    chosenPartitions.append(P[-2]) #append final partition
    chosenPartitions.append(Pgoal)

    circumstances = chosenPercentages.copy()
    circumstances.append(1)
    circumstances.append(1.1)

    # Plot the partitions
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(chosenPartitions)-1):
        Ry = [circumstances[i]] * len(chosenPartitions[i])
        ax.scatter(chosenPartitions[i], Ry, color='blue', s=50)
    Ry = [circumstances[len(chosenPartitions)-1]] * len(chosenPartitions[len(chosenPartitions)-1])
    ax.scatter(chosenPartitions[len(chosenPartitions)-1], Ry, color='red', s=50)
    # Set axis labels and title
    ax.set_xlabel('Partitions at different stages')
    ax.set_ylabel('Progression 0% -> 100%')
    ax.set_title('Progression of Estimated Partition')

    plt.show()     

#progression(s1, Gammas.unif(), GD, objG)

def variationGammaEta(S, mode, chosenEta = np.linspace(0, 50, 11), Fully = 0, algorithm = GD, obj = objG):
    """
    GammaSet consists of the gamma distributions to be compared such as:  [Gammas.Eunif(), Gammas.normC(), ..]
    FirstFully: 0, 1, decided whether we only take the first gamma from the GammaSet and analyse it in both graphs with the partitions and optimal elements, or we only look at the optimal decisions and optimal risk partitions of all gammas in GammaSet.
    """
    a = S.a
    b = S.b
    fig1, Rax = plt.subplots(figsize=(8, 6))
    fig2, Dax = plt.subplots(figsize=(8, 6))    
    
    # Define the chosen values for eta

    circumstances = chosenEta.copy()

    GammaSet, Labels, Colors = GamLabCol(mode)

    for i, gamma in enumerate(GammaSet):
        S.Gamma = gamma
        print(i, Labels[i])

        Rpartitions = [] #containing solutions without boundaries
        decisions = [] #containing the decisions

        for eta in chosenEta:
            S.eta = eta
            P = algorithm(S, obj)
            #IntrinsicComparison(S, algorithm, obj, P=P)
            Rpartitions.append(P[1:-1])
            decisions.append([optimalDecision(S, P[j],P[j+1]) for j in range(len(P)-1)])

        Rpartitions = np.array(Rpartitions)
        decisions = np.fliplr(np.array(decisions)) #flipping decisions

        #we scatter the first eta-partitions to introduce the Labels on the agenda:
        Ry = [circumstances[0]] * len(Rpartitions[0])
        optDz = [circumstances[0]] * len(decisions[0])

        Rax.scatter(Rpartitions[0], Ry, color=Colors[i], s=50, label=Labels[i])
        Dax.scatter(decisions[0],optDz,marker='*' ,color=Colors[i], s=50, label=Labels[i])

        #And the boundaries
        Rax.scatter([a,b], [circumstances[0]]*2, color='black', s=50)
        Dax.scatter([trans(S, a),trans(S, b)], [circumstances[0]]*2, color='black', s=50)
        
        #plot the remaining etas
        for j in range(1, len(chosenEta)):
            Py = [circumstances[j]] * len(Rpartitions[j])
            optDz = [circumstances[j]] * len(decisions[j])

            Rax.scatter(Rpartitions[j], Py, color=Colors[i], s=50)
            Dax.scatter(decisions[j],optDz,marker='*' ,color=Colors[i], s=50)

            #boundaries for all eta
            Rax.scatter([a,b], [circumstances[j]]*2, color='black', s=50)
            Dax.scatter([trans(S, b),trans(S, a)], [circumstances[j]]*2, color='black', s=50)

        if Fully == 1:
            Dpartitions = trans(S, Rpartitions)
            Gdecisions =  trans(S, decisions)

            Pz = [circumstances[0]] * len(Gdecisions[0]) # y-coords of Gopt of optimal decisions
            Dy = [circumstances[0]] * len(Dpartitions[0]) # y-coords Dopt of optimal risk partitions

            #Colors[i] = 'red'

            Dax.scatter(Dpartitions[0], Dy, color=Colors[i], s=50)
            Rax.scatter(Gdecisions[0],Pz,marker='*' ,color=Colors[i], s=50)    

            for j in range(1, len(chosenEta)):
                    Pz = [circumstances[j]] * len(Gdecisions[j])
                    Rax.scatter(Gdecisions[j],Pz,marker='*' ,color=Colors[i], s=50) #add Gopt of optimal decision 

                    Dy = [circumstances[j]] * len(Dpartitions[j])
                    Dax.scatter(Dpartitions[j], Dy, color=Colors[i], s=50) #add decision partition 
        #S.eta = 1
        #Dax.scatter(optimalDecision(S),1, marker='*', s=150, color='red')
        #Rax.scatter(trans(S,optimalDecision(S)),1, marker='*', s=150, color='red')      
    Rax.set_xlabel('Partitions')
    Dax.set_xlabel('Decisions')
    Rax.set_ylabel('Eta')
    Dax.set_ylabel('Eta')
    Rax.legend()
    Dax.legend()
    plt.show()

variationGammaEta(s1, 4, np.linspace(0,10,3), Fully=1, algorithm=GD)

      
def costOptimum(S, n, F, mode, compare = 0, algorithm = GD, obj = objG):
    """
    n: compare for m=1,..,n objective values
    f: cost function
    compare = 1,2,3 : compare gammas, etas, cost functions resp.
    """

    
    GammaSet, Labels, Colors = GamLabCol(mode)
    Etas = np.linspace(1,19,3)
    gamma = GammaSet[0]
    eta = S.eta
    optimum = 3
    indicator = np.where(compare==1, len(GammaSet),0)+np.where(compare==2, len(Etas),0) + np.where(compare==3, len(F),0)
    vary = np.linspace(-.01,.01,indicator)
    if mode ==1 and compare ==1:
        vary = np.zeros(len(GammaSet))
    
    fig1, ax1 = plt.subplots() #plot for costs at m =1,..n, where that decision is optimal
    fig2, ax2 = plt.subplots() #plot differences obj function and cost function and see intersection

    def optCostPlot(objlist, compare, f = F[0], gamma=gamma, eta = eta):
        c = [(objlist[i+1]-objlist[i])/(f(i+2)-f(i+1)) for i in range(len(objlist)-1)]
        print(c)
        X = [x for x in range(2,n)]
        Y = [[c[i],c[i+1]] for i in range(n-2)]
        amount = 0
        for x, (y1, y2) in zip(X, Y):

            if compare ==1:
                if amount == 0:
                    ax1.plot([x+vary[i]] * 2, [y1, y2], '-o', color=Colors[i], label = Labels[i])
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o', color=Colors[i])
                amount +=1 
            if compare ==2:
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o', label= f'eta = {eta}')
            if compare ==3:
                ax1.plot([x+vary[i]] * 2, [y1, y2], '-o',label = costlabel(f))

        cDiff = np.diff(c)
        cPer = [cDiff[i]/c[i] for i in range(len(c)-1)] 
        return c

    def plotobj(objlist, compare, gamma=gamma, eta=S.eta):

        X = np.arange(2,n+1) #values n = 1,...,n-1
        diffObj = np.diff(np.array(objlist))
        if compare == 1:
            ax2.plot(X,diffObj, color=Colors[i], label = Labels[i])
        if compare == 2:
            ax2.plot(X,diffObj, label = f'eta = {S.eta}') #with corresponding objective difference values 1-2, 2-3, .. n-1-n.
        if compare == 3:
            ax2.plot(X,diffObj, color=Colors[0], label = Labels[0])

    def plotcost(optimum, c, f=F[0]):
        #plots the cost curves against an objective function with corresponding cost array c
        #n = i + 1 optimal for c in c[i], c[i+1]

        cfactor = .5*(c[optimum-1] +c[optimum-2]) 
        Nvalues = np.arange(1,n+1) #values n = 1,...,n
        costdifferences = cfactor * np.diff(f(np.array(Nvalues)))
        X = np.arange(2,n+1) #values n = 1,...,n-1
        ax2.plot(X, costdifferences, label = f'{costlabel(f)}, costfactor {cfactor:.5f}, optimum at {optimum}')
        if compare == 1:
            ax1.scatter(optimum + vary[i], cfactor, label = f'{Labels[i]}', s =200)

        if compare == 2:
            ax1.scatter(optimum + vary[i], cfactor, label = f'eta={eta}', s =200)

        if compare == 3:
            ax1.scatter(optimum + vary[i], cfactor, label = f'{Labels[0]}', s =200)    


    if compare == 1:
        amount = 0
        for i, gamma in enumerate(GammaSet):
            print(Labels[i])
            eta = S.eta
            S.Gamma = gamma
            objList=np.zeros(n)
            for m in range(1,n+1):
                S.n = m
                P = algorithm(S, obj)
                value = invUsp(S, goal_function(S,P))
                objList[m-1] = value
                print(f'{m}, {value}')
            plotobj(objList, compare =1, eta = eta)
            if amount == 0:
                plotcost(optimum, optCostPlot(objlist=objList, compare=1,f=F[0], gamma=gamma, eta= eta))
            amount += 1

    if compare == 2:
        amount = 0
        S.Gamma = GammaSet[0]
        gamma =  S.Gamma
        for i, eta in enumerate(Etas):
            S.eta = eta
            print('eta:', Etas[i])

            objList=np.zeros(n)
            for m in range(1,n+1):
                S.n = m
                P = algorithm(S, obj)
                goal =  goal_function(S,P)
                value = invUsp(S, goal)
                objList[m-1] = value
                print(f'{m}, {value}')
            plotobj(objList, compare =2)
            if amount ==0:
                plotcost(optimum, optCostPlot(objlist=objList, compare=2,f=F[0], gamma=gamma, eta= S.eta))
            amount += 1
    if compare == 3:
        S.Gamma = GammaSet[0]
        gamma =  S.Gamma
        objList=np.zeros(n)
        for m in range(1,n+1):
            S.n = m
            P = algorithm(S, obj)
            value = invUsp(S, goal_function(S,P))
            objList[m-1] = value
            print(f'{m}, {value}')
        plotobj(objList, compare =3)
        for i, f in enumerate(F):
            plotcost(optimum, optCostPlot(objlist=objList, compare=3,f=f, gamma=gamma, eta= eta),f)
    ax1.legend()
    ax2.legend()

    plt.show()
    return 
F = [lin, log, quad, exp]
#costOptimum(s1, n =5, F =[quad], mode =55, compare=1, algorithm=GD)


 