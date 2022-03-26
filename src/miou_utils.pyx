from __future__ import division

cimport cython
cimport numpy as np
import numpy as np

def fast_cm(unsigned char[::1] preds, unsigned char[::1] gt,
            int n_classes):
    cdef np.ndarray[np.int_t, ndim=2] cm = np.zeros((n_classes, n_classes), dtype=np.int_)
    cdef np.intp_t i,a,p, n = gt.shape[0]

    for i in range(n):
        a = gt[i]
        p = preds[i]
        cm[a, p] += 1
    return cm

    cdef unsigned int pi = 0
    cdef unsigned int gi = 0
    cdef unsigned int ii = 0
    cdef unsigned int denom = 0
    cdef unsigned int n_classes = cm.shape[0]
    cdef np.ndarray[np.float_t, ndim=1] IU = np.ones(n_classes)
    cdef np.intp_t i
    for i in xrange(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU
