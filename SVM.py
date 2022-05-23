import cvxpy as cp

def ERM_SVM(ξ):

    X, Y = ξ[0], ξ[1]
    d, N = X.shape

    θ = cp.Variable(shape = (d+1, 1))
    loss = cp.Variable(shape = (N, 1)) # Models \loss^\epsilon (i)

    objective = cp.Minimize(1/N * cp.sum(loss))

    nonnegativity_constraints = []
    loss_constraints = []

    for i in range(0, N):
        nonnegativity_constraints.append(loss[i] >= 0)
        loss_constraints.append(loss[i] >= 1 - Y[i]*(θ[0:d].T@X[:, i] - θ[-1]))

    complete_constraints = nonnegativity_constraints + loss_constraints

    model = cp.Problem(
                    objective=objective,
                    constraints=complete_constraints
                    )

    model.solve()

    return θ.value, model.value

def HR_SVM(ξ, 
           α, 
           ϵ, 
           r):

    # This part here will depend on the type of the problem. (nature of ξ and θ)------
    X, Y = ξ[0], ξ[1]
    d, N = X.shape

    θ = cp.Variable(shape = (d+1, 1))
    loss = cp.Variable(shape = (N, 1)) # Models \loss^\epsilon (i)
    w = cp.Variable(shape = (N, 1))

    λ = cp.Variable(nonneg = True, shape = (1, ))
    β = cp.Variable(nonneg = True, shape = (1, ))

    η = cp.Variable()
    W = cp.Variable() # Models worst case, here max \loss^\epsilon (i)

    objective = cp.Minimize(1/N*cp.sum(w) + (r-1)*λ + α*β + η)

    nonnegativity_constraints = []
    soc_constraints = []

    # Loss definition-------------------------
    # SVM loss = max{1-Y'(w'X-b)}, loss_epsilon = max{1-Y'(w'X-b) + ϵ |w|_2^2/|w|}
    # Second order cone constraints and non-negativity constraints
    for i in range(0, N):
        soc_constraints.append(cp.SOC(cp.reshape(loss[i] - 1 + Y[i]*(θ[0:d].T@X[:, i] - θ[-1]), ()), ϵ*θ))
        nonnegativity_constraints.append(loss[i] >= 0)
    
    # ----------------------------------------
    # Dual constraints, indep of loss
    exc_constraints = []
    worst_case_constraints = []

    for i in range(0, N):
        exc_constraints.append(cp.constraints.exponential.ExpCone(-1*w[i], λ, η - loss[i]))
        worst_case_constraints.append(W >= loss[i])
        exc_constraints.append(cp.constraints.exponential.ExpCone(-1*w[i], λ, (η - W + β)))

    # Combining constraints to a single list
    complete_constraints = nonnegativity_constraints + soc_constraints + exc_constraints + worst_case_constraints

    # Problem definition
    model = cp.Problem(
                    objective=objective,
                    constraints=complete_constraints)

    model.solve()

    return θ.value, model.value