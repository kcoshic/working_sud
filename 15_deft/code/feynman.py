from __future__ import division
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, diags, spdiags

import deft_core

# Calculate correction of log_Z at t_infty (MaxEnt) using Feynman diagrams
def run_t_infty(phi_infty, Delta, N):
    G = len(phi_infty)
    
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diag = np.linalg.eigh(Delta_mat)
    
    #e_val0 = Delta_diag[0][0]
    e_vec0 = sp.array(Delta_diag[1][:,0]).ravel()
    #e_val1 = Delta_diag[0][1]
    e_vec1 = sp.array(Delta_diag[1][:,1]).ravel()
    #e_val2 = Delta_diag[0][2]
    e_vec2 = sp.array(Delta_diag[1][:,2]).ravel()    

    K_mat = diags(sp.exp(-phi_infty),0).todense() * (N/G)

    K_proj_mat = sp.mat(sp.zeros((3,3)))
    for i in range(G):
        for j in range(G):
            K_proj_mat[0,0] += K_mat[i,j] * e_vec0[i] * e_vec0[j]
            K_proj_mat[0,1] += K_mat[i,j] * e_vec0[i] * e_vec1[j]
            K_proj_mat[0,2] += K_mat[i,j] * e_vec0[i] * e_vec2[j]
            K_proj_mat[1,1] += K_mat[i,j] * e_vec1[i] * e_vec1[j]
            K_proj_mat[1,2] += K_mat[i,j] * e_vec1[i] * e_vec2[j]
            K_proj_mat[2,2] += K_mat[i,j] * e_vec2[i] * e_vec2[j]
    K_proj_mat[1,0] = K_proj_mat[0,1]
    K_proj_mat[2,0] = K_proj_mat[0,2]
    K_proj_mat[2,1] = K_proj_mat[1,2]

    K_proj_inv = sp.mat(sp.linalg.inv(K_proj_mat))

    P_mat = sp.mat(sp.zeros((G,G)))
    for i in range(G):
        for j in range(G):
            P_mat[i,j] = K_proj_inv[0,0] * e_vec0[i] * e_vec0[j] \
                       + K_proj_inv[0,1] * e_vec0[i] * e_vec1[j] \
                       + K_proj_inv[0,2] * e_vec0[i] * e_vec2[j] \
                       + K_proj_inv[1,0] * e_vec1[i] * e_vec0[j] \
                       + K_proj_inv[1,1] * e_vec1[i] * e_vec1[j] \
                       + K_proj_inv[1,2] * e_vec1[i] * e_vec2[j] \
                       + K_proj_inv[2,0] * e_vec2[i] * e_vec0[j] \
                       + K_proj_inv[2,1] * e_vec2[i] * e_vec1[j] \
                       + K_proj_inv[2,2] * e_vec2[i] * e_vec2[j] 

    V = sp.exp(-phi_infty) * (N/G)

    Z_correction_1st = diagrams_1st_order(G, P_mat, V)

    return Z_correction_1st

# Calculate correction of log_Z at finite t using Feynman diagrams
def run_t_finite(phi_t, R, Delta, t, N):
    G = len(phi_t)
    # Propagator
    H = deft_core.hessian(phi_t, R, Delta, t, N, regularized=False)
    A_mat = H.todense() * (N/G)
    P_mat = np.linalg.inv(A_mat)
    # Vertex
    V = sp.exp(-phi_t) * (N/G)

    Z_correction_1st = diagrams_1st_order(G, P_mat, V)

    return Z_correction_1st

# Feynman diagrams of order N^{-1}
def diagrams_1st_order(G, P, V):
    ### 1
    s1 = 0.0
    for i in range(G):
        s1 += V[i]*P[i,i]**2
    s1 = (-1) * s1/8
    ### 2
    s2 = 0.0
    for i in range(G):
        for j in range(G):
            s2 += V[i]*V[j]*P[i,i]*P[i,j]*P[j,j]
    s2 = s2/8
    ### 3
    s3 = 0.0
    for i in range(G):
        for j in range(G):
            s3 += V[i]*V[j]*P[i,j]**3
    s3 = s3/12

    s_tot = s1 + s2 + s3

    return s1, s2, s3, s_tot
