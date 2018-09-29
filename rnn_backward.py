import numpy as np

def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    2. a_next = tanh(Wax dot x + Waa dot a_prev + b)
    Formula for previous derivatuve
        2.1     dtanh = (1 - tanh)^2
        2.2     dWax = dtanh dot xt.T
        2.3     dWaa = dtanh dot a_prev.T
        2.4     dba = sum dtanh
        2.5     dxt = Wax.T dot dtanh
        2.6     da_prev = Waa.T dot dtanh

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute the gradient of tanh with respect to a_next
    dtanh = da_next * (1 - np.square(a_next))

    # compute the gradient of the loss with respect to x
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to a
    da_pev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    # compute the gradient with respect to b
    dba = np.sum(dtanh, axis = 1, keepdims = True)

    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
    dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
    da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
    dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
    dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
    dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # Initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t.
        gradients = rnn_cell_backward(da[:,:,t] + da_prevt, caches[t])
        dx[:, :, t] = gradients["dxt"]
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]
        da_prevt = gradients["da_prev"]

    # Set da0 to the gradient of a which has been backpropagated through all time-steps
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}

    return gradients
