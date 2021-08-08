def report_loss(X, y, coefficients, loss_function, iteration):
    if iteration % 100 == 0:
        loss = loss_function(X, y, coefficients)
        print(f"Loss at iteration {iteration}: {loss}")
    return


def gradient_descent(
    X, y, coefficients, loss_function, gradient, learning_rate, max_iterations
):
    iteration = 0
    while iteration < max_iterations:
        change = learning_rate * gradient(X, y, coefficients)
        coefficients -= change
        report_loss(X, y, coefficients, loss_function, iteration)
        iteration += 1
    return coefficients
