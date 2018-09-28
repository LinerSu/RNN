import numpy as np

def tanh(x):
    """
    Compute the Tanh function for the input.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    th -- Tanh(x)
    """
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    th = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    return th

def tanh_grad(x):
    """
    Compute the gradient for the Tanh function.
    Tanh'(x) = 1 - Tanh(x)^2

    Arguments:
    x -- A scalar or numpy array.

    Return:
    dth -- Your computed gradient.
    """
    dth = 1 - np.square(tanh(x))
    return dth

def test_tanh():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running tanh & tanh_grad tests..."
    x = np.array([
        [ 0.41287266, 0,  0.78215209],
        [ 0.76983443,  0.46052273,  0.4283139 ],
        [-0.18905708,  0.57197116,  0.53226954]])
    f = tanh(x)
    g = tanh_grad(x)
    print f
    f_ans = np.array([
        [0.39090909, 0, 0.65394021],
        [0.64683316,  0.43051015,  0.40391124],
        [-0.18683636,  0.51680543,  0.48711402]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.84719008, 1.,    0.5723622 ],
        [0.58160686, 0.81466101, 0.8368557 ],
        [0.96509217, 0.73291214, 0.76271992]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "Pass all tests..."


if __name__ == "__main__":
    test_tanh()
