#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import time
import sys
import numpy as np
import faiss
from datasets import ivecs_read

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

k = 1000

print("load data")

xb = mmap_bvecs('bigann/bigann_base.bvecs')
xq = mmap_bvecs('bigann/bigann_query.bvecs')
xt = mmap_bvecs('bigann/bigann_learn.bvecs')
gt = ivecs_read('bigann/gnd/idx_1000M.ivecs')

nq, d = xq.shape

print("Testing IVF Flat with HNSW quantizer")
# Param of PQ
M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
# Param of IVF
nlist = 31622  # The number of cells (space partition). Typical value is sqrt(N)
# Param of HNSW
hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
index.cp.min_points_per_centroid = 5   # quiet warning
index.quantizer_trains_alone = 2

# to see progress
index.verbose = True

print("training")
xt = xt.astype(np.float32)
index.train(xt)

print("add")
xb_batch = 0
xb_step = 1000000
while xb_batch < 1000:
    print("add %dth batch of vector" %(xb_batch))
    xb_pos = xb_batch * xb_step
    xb_slice = xb[xb_pos:xb_pos + xb_step]
    xbf_slice = []
    for i in range(0, xb_step):
        xbf_slice.append(xb_slice[i]*1.0)
    index.add(xbf_slice)
    xb_batch = xb_batch + 1

print("search")
quantizer.hnsw.efSearch = 64
for nprobe in 1, 4, 16, 64, 256:
    print("nprobe", nprobe, end=' ')
    index.nprobe = nprobe
    evaluate(index)

def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    xq = xq.astype(np.float32)
    D, I = index.search(xq, k)
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))


