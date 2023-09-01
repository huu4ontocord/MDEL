import os

import numpy as np


def is_contiguous(arr):
    start = None
    prev = None
    contiguous = True

    for idx in arr:
        if start is None:
            start = idx
        if prev is None or idx == prev + 1:
            prev = idx

            continue

        contiguous = False

        break

    return contiguous, start, idx + 1


def np_memmap(file_name, data=None, idxs=None, shape=None, dtype=np.float32, offset=0, order='C'):
    if not file_name.endswith('.mmap'):
        file_name += '.mmap'
        
    if os.path.exists(file_name):
        mode = 'r+' 
    else:
        mode = 'w+'

    if shape is None and data is not None:
        shape = data.shape
        
    if not shape:
        shape = [0, 1]
            
    memmap = np.memmap(file_name, mode=mode, dtype=dtype, shape=tuple(shape), offset=offset, order=order)
    
    if idxs:
        contiguous, start, end = is_contiguous(idxs)
        
    if data is not None:
        if tuple(shape) == tuple(data.shape):
            memmap[:] = data
        elif contiguous:
            memmap[start:end] = data
        else:
            memmap[idxs] = data
            
    return memmap


def get_np_memmap_length(file_name, shape, dtype=np.float32):
    if not os.path.exists(file_name):
        return shape[0]
    
    else:
        size = np.dtype(dtype).itemsize * np.prod(shape[1:])

        return int(os.path.getsize(file_name) / size)
