import numpy as np

"""
    For more information about softmax, please check:
        https://deepnotes.io/softmax-crossentropy
"""

def softmax(x):
    """
    Compute the softmax function for the input.
    Input x.shape == (n_x, m)
    Softmax(x) = Softmax(x + a), use this to decrease exp value

    Arguments:
    x -- A (n_x, m) dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        maxx = np.max(x, axis = 0)
        x = np.exp(x - maxx)
        sumx = np.sum(x, axis = 0)
        x = x / sumx
    else:
        x = x / x

    assert x.shape == orig_shape
    return x


def softmax_grad(x, i):
    """
    Compute the gradient function for the softmax.
    Input x.shape == (n_x, m)

    ds = s_j*((i==j) - s_i)

    Arguments:
    x -- A (n_x, m) dimensional numpy matrix.
    i -- derivative with respect to value by index i

    Return:
    x -- You are allowed to modify x in-place
    """
    s = softmax(x)
    ds = np.multiply(s, s[i])
    ds[i] = s[i] - ds[i]

    return ds

def test_softmax():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running softmax tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([1.,  1.])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,3],[1002,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.26894142],
        [0.73105858, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001],[-1002]]))
    print test3
    ans3 = np.array([[0.73105858], [0.26894142]])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    test4 = softmax(np.array([[-1001,-1002]]))
    print test4
    ans4 = np.array([[1., 1.]])
    assert np.allclose(test4, ans4, rtol=1e-05, atol=1e-06)

    print "Pass all tests..."

def test_softmax_grad():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running softmax_grad tests..."
    print "Pass all tests..."
    test1 = softmax_grad(np.array([[1],[2]]), 0)
    print test1
    ans1 = np.array([[0.19661193], [0.19661193]])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    print "Pass all tests..."

if __name__ == "__main__":
    test_softmax()
    test_softmax_grad()
