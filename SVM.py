import picos as pic


def ERM_svm(ξ):

    model = pic.Problem()

    # This part here will depend on the type of the problem. (nature of ξ and θ)------
    X, Y = ξ[0], ξ[1]
    N, d = X.shape

    # Decision Variables
    θ = pic.RealVariable("theta", (d+1, 1))
    loss = pic.RealVariable("loss", (N, 1))  # Models \loss^\epsilon (i)

    # Objective Function
    model.set_objective('min', 1/N*pic.sum(loss))

    # Loss definition-------------------------
    # SVM loss = max{1-Y'(w'X-b)}, loss_epsilon = max{1-Y'(w'X-b) + ϵ |w|_2^2/|w|}
    for i in range(0, N):
        model.add_constraint(loss[i] >= 0)
        model.add_constraint(loss[i] >= 1 - Y[i]*(θ[0:d].T*X[i, :] - θ[-1]))
    # ----------------------------------------

    model.solve(solver='mosek')

    return θ.value, model.value
