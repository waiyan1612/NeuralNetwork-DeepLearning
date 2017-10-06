from builtins import range
import numpy as np
import math


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x >= 0) * dout
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.random_sample(x.shape) > p
        dropped = x.size - np.sum(mask)
        print("Dropped neurons:", dropped, "/", x.size, "(", dropped/x.size, ")")
        out = np.multiply(x, mask) / (1-p)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = np.multiply(dout, mask) / (1-dropout_param['p'])
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction.
         We will use the same definition in lecture notes 3b, slide 13 (ie. same padding on both sides).

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + pad - HH) / stride
      W' = 1 + (W + pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    p = int(pad/2)
    
    oh = 1 + int((x.shape[2] + pad - w.shape[2]) / stride)
    ow = 1 + int((x.shape[3] + pad - w.shape[3]) / stride)
    
    f, c, hh, ww = w.shape
    n = c * hh * ww
    m = x.shape[0]
    out = np.empty(shape=(m, f, oh, ow))
    
    # padding input
    padded_x = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), 'constant', constant_values=(0))
    reshaped_x = np.empty(shape=(m, oh*ow, n))

    # transform w
    transformed_w = w.reshape(f, -1)

    # transform x
    for k in range(m):
        transformed_x = np.empty(shape=(oh*ow, n))
        index = i = 0
        for row in range(oh):
            j = 0
            for col in range(ow):
                transformed_x[index] = padded_x[k, :, i:i+hh, j:j+ww].reshape(-1, n)
                j += stride
                index += 1
            i += stride
        reshaped_x[k] = transformed_x
        out[k] = (transformed_w.dot(transformed_x.T) + b[:,None]).reshape(-1, oh, ow)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, reshaped_x)
    return out, cache


def conv_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x of shape (N, C, H, W)
    - dw: Gradient with respect to w of shape (F, C, HH, WW)
    - db: Gradient with respect to b of shape (F, )
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, _, _, reshaped_x = cache
    m, f, _, _ = dout.shape
    _, c, hh, ww = w.shape
    stride = cache[3]['stride']
    pad = cache[3]['pad']
    p = int(pad/2)
    
    oh = 1 + int((x.shape[2] + pad - w.shape[2]) / stride)
    ow = 1 + int((x.shape[3] + pad - w.shape[3]) / stride)
    
    db = dout.sum(axis=(0,2,3))
    dw = np.tensordot(dout.transpose(1, 2, 3, 0).reshape(f, -1, m), reshaped_x, axes=([1,2],[1,0])).reshape(w.shape)
    dx = np.dot(w.reshape(f, -1).T, dout.transpose(1, 0, 2, 3).reshape(f, -1))
    
    index = 0
    dx_padded = np.zeros(shape=(m, c, x.shape[2]+pad, x.shape[3]+pad))
    for k in range(m):
        i = 0
        for i in range(oh):
            j = 0
            for j in range(ow):
                dx_padded[k,:,i:i+hh,j:j+ww] += dx[:,index].reshape(c,hh,ww)
                j += stride
                index += 1
            i += stride
    dx = dx_padded[:,:,p:-p,p:-p]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    Forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    oh = int((x.shape[2] - pool_height)/stride) + 1
    ow = int((x.shape[3] - pool_width)/stride) + 1
    out = np.empty(shape=(x.shape[0], x.shape[1], oh, ow))

    max_fields = np.empty(shape=(oh * ow * x.shape[0] * x.shape[1], 2), dtype=int)
    index = 0
    for k in range(x.shape[0]):
        for c in range(x.shape[1]):
            i = 0
            for row in range(pool_height, x.shape[2]+1, stride):
                j = 0
                for col in range(pool_width, x.shape[3]+1, stride):
                    tmp = x[k, c, row-pool_height:row, col-pool_width:col]
                    max_i, max_j = np.unravel_index(tmp.argmax(), tmp.shape)
                    out[k, c, i, j] = x[k, c, max_i + row - pool_height , max_j + col - pool_width]
                    max_fields[index] = (max_i, max_j) 
                    j += 1
                    index += 1
                i += 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, max_fields)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param, max_fields = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    oh = int((x.shape[2] - pool_height)/stride) + 1
    ow = int((x.shape[3] - pool_width)/stride) + 1
    dx = np.zeros(shape=(x.shape))
    index = 0
    for k in range(x.shape[0]):
        for c in range(x.shape[1]):
            i = 0
            for row in range(pool_height, x.shape[2]+1, stride):
                j = 0
                for col in range(pool_width, x.shape[3]+1, stride):
                    #get indices
                    max_i, max_j = max_fields[index]
                    field = np.zeros(shape=(pool_height, pool_width))
                    field[max_i, max_j] = 1
                    field *= dout[k, c, i, j]
                    dx[k, c, row-pool_height:row, col-pool_width:col] += field
                    j += 1
                    index += 1
                i += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
