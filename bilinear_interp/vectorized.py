import numpy as np
from numpy import int64


def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # get axis size from ndarray shape
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1
    res = np.empty((N,H2,W2,C), dtype=int64)
    useX = np.empty((N,H2,W2,2),dtype=int64)
    useY = np.empty((N,H2,W2,2),dtype=int64)
    valueOfIntX = np.empty((N,H2,W2),dtype=int64)
    valueOfIntX = b[...,0].copy()
    x = valueOfIntX
    valueOfIntY = np.empty((N,H2,W2),dtype=int64)
    valueOfIntY = b[...,1].copy()
    y = valueOfIntY
    cx = np.floor(valueOfIntX).astype(np.int)
    cy = np.floor(valueOfIntY).astype(np.int)
    cxplus = cx+1
    cyplus = cy+1
    cxplus2 = cxplus+1
    cyplus2 = cyplus+1
    extenda = np.pad(a,1)
    f1 = np.empty((N,H2,W2,C),dtype=np.float64)
    f2 = np.empty((N,H2,W2,C),dtype=np.float64)
    f3 = np.empty((N,H2,W2,C),dtype=np.float64)
    f4 = np.empty((N,H2,W2,C),dtype=np.float64)
    for i in range(N):
        f1[i,:,:,:] = extenda[i+1,cxplus[i],cyplus[i],1:C+1]
    for i in range(N):
        f2[i,:,:,:] = extenda[i+1,cxplus2[i],cyplus[i],1:C+1]
    for i in range(N):
        f3[i,:,:,:] = extenda[i+1,cxplus[i],cyplus2[i],1:C+1]
    for i in range(N):
        f4[i,:,:,:] = extenda[i+1,cxplus2[i],cyplus2[i],1:C+1]
    a = cx+1-x
    b = x - cx
    c = cy+1-y
    d = y-cy
    aplus = np.tile(a,(C,1,1,1))
    aplus = aplus.transpose(1,2,3,0)
    bplus = np.tile(b,(C,1,1,1))
    bplus = bplus.transpose(1,2,3,0)
    cplus = np.tile(c,(C,1,1,1))
    cplus = cplus.transpose(1,2,3,0)
    dplus = np.tile(d,(C,1,1,1))
    dplus = dplus.transpose(1,2,3,0)
    af1 = np.empty((N,H2,W2,C),dtype=np.float64)
    bf2 = np.empty((N,H2,W2,C),dtype=np.float64)
    af3 = np.empty((N,H2,W2,C),dtype=np.float64)
    bf4 = np.empty((N,H2,W2,C),dtype=np.float64)
    af1 = aplus*f1
    bf2 = bplus*f2
    af3 = aplus*f3
    bf4 = bplus*f4
    fx1 = af1+bf2
    fx2 = af3+bf4
    cfx1 = np.empty((N,H2,W2,C),dtype=np.float64)
    dfx2 = np.empty((N,H2,W2,C),dtype=np.float64)
    cfx1 = cplus*fx1
    dfx2 = dplus*fx2
    fres = cfx1+dfx2
    res = fres.reshape(N,H2,W2,C)
    res = res.astype(int64)
    # TODO: Implement vectorized bilinear interpolation
    return res