#import scipy as sp
#import numpy as np
import new_numpy as np
import new_scipy as sp
from scipy.sparse import csr_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import det, eigh, solve, eigvalsh, inv
import scipy.optimize as opt

import utils
import laplacian
import maxent
import time
import feynman

# Put hard bounds on how big or small t can be
# T_MIN especially seems to help convergence
T_MAX = 20
T_MIN = -20
LOG_E_RANGE = 100
PHI_MAX = utils.PHI_MAX
PHI_MIN = utils.PHI_MIN
MAX_DS = -1E-3
#MAX_DS = -1E-3
PHI_STD_REG = utils.PHI_STD_REG 

class Results(): pass;

# Represents a point along the MAP curve
class MAP_curve_point:

    def __init__(self, t, Q, log_E, details=False):
        self.t = t
        self.Q = Q
        self.phi = utils.prob_to_field(Q)
        self.log_E = log_E
        #self.details = details

# Represents the MAP curve
class MAP_curve:

    def __init__(self):
        self.points = []
        self._is_sorted = False

    def add_point(self, t, Q, log_E, details=False):
        #print 'Added point at t==%f'%t
        point = MAP_curve_point(t, Q, log_E, details)
        self.points.append(point)
        self._is_sorted = False

    def sort(self):
        self.points.sort(key=lambda x: x.t)
        self._is_sorted = True

    # Use this to get actual points along the MAP curve.
    # Ensures that points are sorted
    def get_points(self):
        if not self._is_sorted:
            self.sort()
        return self.points

    def get_maxent_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[0]
        assert(p.t == -sp.Inf)
        return p

    def get_histogram_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[-1]
        assert(p.t == sp.Inf)
        return p

    def get_log_evidence_ratios(self, finite=True):
        log_Es = sp.array([p.log_E for p in self.points])
        ts = sp.array([p.t for p in self.points])

        if finite:
            indices = (log_Es > -np.Inf)*(ts > -np.Inf)*(ts < np.Inf)
            return log_Es[indices], ts[indices]
        else:
            return log_Es, ts


# Convention: action, gradient, and hessian are G/N * the actual.
# This provides for more robust numerics

# Evaluate the action of a field given smoothness criteria
def action(phi, R, Delta, t, N, phi_in_kernel=False, regularized=False):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    ones_col = sp.mat(sp.ones(int(G))).T

    if phi_in_kernel:
        S_mat = G*R_col.T*phi_col + G*ones_col.T*quasiQ_col
    else:
        S_mat = 0.5*sp.exp(-t)*phi_col.T*Delta_sparse*phi_col \
           + G*R_col.T*phi_col \
           + G*ones_col.T*quasiQ_col

    if regularized:
        S_mat += 0.5*(phi_col.T*phi_col)/(N*PHI_STD_REG**2)

    S = S_mat[0,0]
    assert np.isreal(S)
    return S

# Evaluate action gradient w.r.t. a field given smoothness criteria
def gradient(phi, R, Delta, t, N, regularized=False):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    grad_col = sp.exp(-t)*Delta_sparse*phi_col + G*R_col - G*quasiQ_col

    if regularized:
        grad_col += phi_col/(N*PHI_STD_REG**2)

    grad = sp.array(grad_col).ravel()

    assert all(np.isreal(grad))
    return grad

# Evaluate action hessain w.r.t. a field given smoothness criteria
# NOTE: returns sparse matrix, not dense matrix!
def hessian(phi, R, Delta, t, N, regularized=False):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    Delta_sparse = Delta.get_sparse_matrix()
    H = sp.exp(-t)*Delta_sparse + G*diags(quasiQ,0)

    if regularized:
        H += diags(np.ones(int(G)),0)/(N*PHI_STD_REG**2)

    return H

# Get log ptgd of the maxent density
def log_ptgd_at_maxent(phi_M, R, Delta, N):
    kernel_dim = Delta._kernel_dim
    M = utils.field_to_prob(phi_M)
    M_on_kernel = sp.zeros([kernel_dim, kernel_dim])
    kernel_basis = Delta._kernel_basis
    lambdas = Delta._eigenvalues
    for a in range(int(kernel_dim)):
        for b in range(int(kernel_dim)):
            psi_a = sp.ravel(kernel_basis[:,a])
            psi_b = sp.ravel(kernel_basis[:,b])
            M_on_kernel[a,b] = sp.sum(psi_a*psi_b*M)

    # Compute log occam factor at infinity
    log_Occam_at_infty = -0.5*sp.log(det(M_on_kernel)) \
                         - 0.5*sp.sum(sp.log(lambdas[kernel_dim:]))

    assert np.isreal(log_Occam_at_infty)

    # Compute the log likelihod at infinty
    log_likelihood_at_infty = -N*sp.sum(phi_M*R) - N

    assert np.isreal(log_likelihood_at_infty)

    # Compute the log posterior (not sure this is right)
    log_ptgd_at_maxent = log_likelihood_at_infty + log_Occam_at_infty

    assert np.isreal(log_ptgd_at_maxent)
    return log_ptgd_at_maxent

# Computes the log of ptgd at t
def log_ptgd(phi, R, Delta, t, N):
    G = 1.*len(phi)
    alpha = 1.*Delta._alpha
    kernel_dim = 1.*Delta._kernel_dim
    H = hessian(phi, R, Delta, t, N)
    H_prime = H.todense()*sp.exp(t)

    S = action(phi, R, Delta, t, N)   
    assert np.isreal(S)

    # First try computing log determinant straight away
    log_det = sp.log(det(H_prime))

    # If result is not finite and real, try computing the sum of eigenvalues,
    # forcing the eigenvalues to be real and nonnegative
    if not (np.isreal(log_det) and np.isfinite(log_det)):
        #print 'Warning: log_det becoming difficult to compute.'
        lambdas = abs(eigvalsh(H_prime))
        log_det = sp.sum(sp.log(lambdas))

    assert (np.isreal(log_det) and np.isfinite(log_det))
    
    # Compute contribution from finite t
    log_ptgd = -(N/G)*S + 0.5*kernel_dim*t - 0.5*log_det

    details = Results()
    details.S = S
    details.N = N
    details.G = G
    details.kernel_dim = kernel_dim
    details.t = t
    details.log_det = log_det
    details.phi = phi

    return log_ptgd, details

# Computes error bars on phi at t
def get_dQ_sq(phi, R, Delta, t, N):
    G = 1.*len(phi)
    Q = utils.field_to_prob(phi)

    # If t is finite, just compute diagonal of covariance matrix
    if np.isfinite(t):

        H = (N/G)*hessian(phi, R, Delta, t, N)

        dQ_sq = np.zeros(int(G))
        for i in range(int(G)):
            delta_vec = np.zeros(int(G))
            delta_vec[i] = 1.0
            v = Q - delta_vec
            a = spsolve(H, v)
            dQ_sq[i] = (Q[i]**2)*np.sum(v*a)

    # If t is not finite, this is a little more sophisticated 
    # but not harder computationally
    else:
        H = (N/G)*spdiags(np.exp(-phi),0,G,G)
        psis = np.mat(Delta._kernel_basis)
        H_tilde = psis.T * H * psis
        H_tilde_inv = inv(H_tilde)
        #dphi_cov = psis*H_tilde_inv*psis.T
        
        dQ_sq = np.zeros(int(G))
        for i in range(int(G)):
            delta_vec = np.zeros(int(G))
            delta_vec[i] = 1.0
            v_col = sp.mat(Q - delta_vec).T
            v_proj = psis.T * v_col
            dQ_sq[i] = (Q[i]**2)*(v_proj.T*H_tilde_inv*v_proj)[0,0]

    # Note: my strange normalization conventions might be causing problmes
    # Might be missing factor of G in here
    return dQ_sq



# Computes predictor step
def compute_predictor_step(phi, R, Delta, t, N, direction, resolution):

    # Make sure direction is just a sign
    assert(direction==1 or direction==-1)

    # Make sure phi is ok
    assert all(phi >= utils.PHI_MIN) 
    assert all(phi <= utils.PHI_MAX)

    # Get current probability dist
    Q = utils.field_to_prob(phi)

    G = 1.*len(Q)

    # Get hessian
    H = hessian(phi, R, Delta, t, N, regularized=False)

    # Comput rho, which indicates direction of step
    rho = G*spsolve(H, Q - R )
    assert all(np.isreal(rho))

    denom = sp.sqrt(sp.sum(rho*Q*rho))
    assert np.isreal(denom)
    assert denom > 0

    # Compute dt based on value of epsilon (the resolution)
    dt = direction*resolution/denom

    # Return phi_new and new t_new
    # WARNING: IT IS NOT YET CLEAR THAT PHI_NEW
    # ISN'T INSANE
    phi_new = phi + rho*dt
    t_new = t + dt
    return phi_new, t_new

# Computes corrector step
def compute_corrector_step(phi, R, Delta, t, N, tollerance=1E-5, report_num_steps=False):

    # Make sure phi_new is ok
    assert all(phi >= utils.PHI_MIN) 
    assert all(phi <= utils.PHI_MAX)

    # Evaluate the probabiltiy distribution
    Q = utils.field_to_prob(phi)

    # Evaluate action
    S = action(phi, R, Delta, t, N, regularized=False)

    # Perform corrector steps until until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:

        # Compute the gradient
        v = gradient(phi, R, Delta, t, N, regularized=False)
        
        # Compute the hessian
        H = hessian(phi, R, Delta, t, N, regularized=False)

        # Solve linear equation to get change in field
        dphi = -spsolve(H,v)

        # Make sure dphi is real and finite
        assert all(np.isreal(dphi))
        assert all(np.isfinite(dphi))        

        # Compute corresponding change in action
        dS = sp.sum(dphi*v)

        # If we're already very close to the max, then dS will be close to zero
        # in this case, we're done already
        if dS > MAX_DS:
            break;

        # Reduce step size until in linear regime
        beta = 1.0
        while True:

            # Make sure beta isn't fucking up
            if beta < 1E-50 :
                print ' --- Something is wrong. ---'
                print 'beta == %f'%beta
                print 'dS == %f'%dS
                print 'S == %f'%S
                print 'S_new == %f'%S_new
                print '|phi| == %f'%np.linalg.norm(phi)
                print '|dphi| == %f'%np.linalg.norm(dphi)
                print '|v| == %f'%np.linalg.norm(v)
                print ''
                assert False

            # Compute new phi 
            phi_new = phi + beta*dphi

            # If new phi is not sane, decrease beta
            if any(phi_new < utils.PHI_MIN) or  any(phi_new > utils.PHI_MAX):
                num_backtracks+=1
                beta *= 0.5 
                continue

            # Compute new action
            S_new = action(phi_new, R, Delta, t, N, regularized=False) # Was True

            # Check for linear regime 
            if (S_new - S <= 0.5*beta*dS):
                break

            # If not in linear regime backtrack value of beta
            else:
                num_backtracks+=1
                beta *= 0.5  
                continue

        # Make sure phi_new is ok
        assert all(phi_new >= utils.PHI_MIN) 
        assert all(phi_new <= utils.PHI_MAX)

        # Comptue new Q
        Q_new = utils.field_to_prob(phi_new)

        # Break out of loop if Q_new is close enough to Q
        gd = utils.geo_dist(Q_new, Q)
        if gd < tollerance:
            break
        
        # Break out of loop with warning if S_new > S. Should not happen,
        # but not fatal if it does. Just means less precision
        # ACTUALLY, THIS SHOULD NEVER HAPPEN!
        elif S_new-S > 0:
            print 'Warning: S_change > 0. Terminating corrector steps.'
            break

        # Otherwise, continue with corrector step
        else:
            # New phi, Q, and S values have already been computed
            phi = phi_new
            Q = Q_new
            S = S_new
            num_corrector_steps += 1

    # After corrector loop has finished, return field
    # Also return stepping stats if requested
    #print 'compute_corrector_step(): tollerance=%e, num_corrector_steps=%d, num_backtracks=%d'%(tollerance,num_corrector_steps,num_backtracks)
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi

# The core algorithm of DEFT, used for both 1D and 2D density esitmation
def compute_map_curve(N, R, Delta, resolution=1E-2, tollerance=1E-3, 
    print_t=False, t_start=0.0, Laplace=True):
    """ Traces the map curve in both directions

    Args:

        R (numpy.narray): 
            The data histogram

        Delta (Smoothness_operator instance): 
            Effectiely defines smoothness

        resolution (float): 
            Specifies max distance between neighboring points on the 
            MAP curve

    Returns:

        map_curve (list): A list of MAP_curve_points

    """

    #resolution=3.14E-2
    #tollerance=1E-3

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    alpha = Delta._alpha
    kernel_basis = Delta.get_kernel_basis()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure the smoothness_operator has the right shape
    assert(G == len(R))

    # Make sure histogram is nonnegative
    assert(all(R >= 0))

    # Make sure that enough elements of counts_array contain data
    assert(sum(R >= 0) > kernel_dim)

    # Inialize map curve
    map_curve = MAP_curve()

    #
    # First compute histogram stuff
    #

    # Get normalied histogram and correpsonding field
    R = R/sum(R)
    phi_0 = utils.prob_to_field(R)
    log_E_R = -np.Inf
    t_R = np.Inf
    map_curve.add_point(t_R, R, log_E_R)

    #
    # Now compute maxent stuff
    #

    # Compute the maxent field and density
    coeffs_infty, phi_infty, success = maxent.compute_maxent_field(R, kernel_basis)
#---
    if not Laplace:
        s1_infty, s2_infty, s3_infty, correction_infty = \
            feynman.run_t_infty(phi_infty, Delta, N)
#---
    # Convert maxent field to probability distribution
    M = utils.field_to_prob(phi_infty)

    # Compute the maxent log_ptgd
    # Important to keep this around to compute log_E at finite t
    log_ptgd_M = log_ptgd_at_maxent(phi_infty, R, Delta, N)

    # This corresponds to a log_E of zero
    log_E_M = 0
    t_M = -sp.Inf
    map_curve.add_point(t_M, M, log_E_M)
    map_curve.coeffs_infty = coeffs_infty
    map_curve.kernel_basis = kernel_basis

    # Set maximum log evidence ratio so far encountered
    log_E_max = -np.Inf #0

    # Compute phi_start by executing a corrector step starting at maxent dist
    phi_start = compute_corrector_step( phi_infty, R, Delta, t_start, N, \
                                        tollerance=tollerance, \
                                        report_num_steps=False)
    
    # Convert starting field to probability distribution
    Q_start = utils.field_to_prob(phi_start)

    # Compute log ptgd
    log_ptgd_start, start_details = log_ptgd(phi_start, R, Delta, t_start, N)

    # Compute corresponding evidence ratio
    log_E_start = log_ptgd_start - log_ptgd_M
#---
    print 'log_E_start'
    print 'log_E lap:', log_E_start
    if not Laplace:
        s1, s2, s3, correction = feynman.run_t_finite(phi_start, R, Delta, t_start, N)
        log_E_start = log_E_start + correction - correction_infty
        print 'log_E new:', log_E_start
        print 't =', t_start
        print s1, s2, s3, '=', correction
        print s1_infty, s2_infty, s3_infty, '=', correction_infty
        print 
#---
    # Adjust max log evidence ratio
    log_E_max = log_E_start if (log_E_start > log_E_max) else log_E_max

    # Set start as first map curve point 
    if print_t:
        print 't == %.2f'%t_start
    map_curve.add_point(t_start, Q_start, log_E_start) #, start_details)

    # Trace map curve in both directions
    for direction in [-1,+1]:
        print '--------------------------'
        if (direction == -1):
            print 'log_E moving toward MaxEnt'
        else:
            print 'log_E moving toward R'
        # Start iteration from central point
        phi = phi_start
        t = t_start
        Q = Q_start
        log_E = log_E_start

        if direction == -1:
            Q_end = M
        else:
            Q_end = R

        # Keep stepping in direction until read the specified endpoint
        while True:

            # Test distance to endpoint
            if utils.geo_dist(Q_end,Q) <= resolution:
                break

            # Take predictor step 
            phi_pre, t_new = compute_predictor_step( phi, R, Delta, t, N, \
                                                     direction=direction, \
                                                     resolution=resolution )

            # If phi_pre is insane, start iterating from phi instead
            if any(phi_pre > PHI_MAX) or any (phi_pre < PHI_MIN):
                phi_pre = phi

            # Compute new distribution
            #Q_pre = utils.field_to_prob(phi_pre)

            #print 'geo_dist(Q_pre,Q) == %f'%utils.geo_dist(Q_pre,Q)

            # Perform corrector stepsf to get new phi
            phi_new = compute_corrector_step( phi_pre, R, Delta, t_new, N, \
                                              tollerance=tollerance, \
                                              report_num_steps=False)

            # Compute new distribution
            Q_new = utils.field_to_prob(phi_new)

            # Print geodistance between Q and Q_new
            #print utils.geo_dist(Q_new,Q)

            # Compute log ptgd
            log_ptgd_new, details_new = log_ptgd(phi_new, R, Delta, t_new, N)

            # Compute corresponding evidence ratio
            log_E_new = log_ptgd_new - log_ptgd_M
#---
            print 'log_E lap:', log_E_new
            if not Laplace:
                s1, s2, s3, correction = feynman.run_t_finite(phi_new, R, Delta, t_new, N)
                log_E_new = log_E_new + correction - correction_infty
                print 'log_E new:', log_E_new
                print 't =', t_new
                print s1, s2, s3, '=', correction
                print s1_infty, s2_infty, s3_infty, '=', correction_infty
                print 
#---
            # Take step
            t = t_new
            Q = Q_new   
            phi = phi_new
            log_E = log_E_new
            details = details_new

            # Add new point to map curve
            if print_t:
                print 't == %.2f'%t
            map_curve.add_point(t, Q, log_E) #, details_new)

            # Adjust max log evidence ratio
            log_E_max = log_E if (log_E > log_E_max) else log_E_max

            # Terminate if log_E is too small. But don't count
            # the t=-inf endpoint when computing log_E_max 
            if (log_E_new < log_E_max - LOG_E_RANGE):
                #print 'Log_E too small. Exiting at t == %f'%t
                break

            #print '\ngeo_dist(Q_new,Q) == %f'%utils.geo_dist(Q_new,Q)  

            # Terminate if t is too large or too small
            if t > T_MAX:
                #print 'Warning: t = %f is too positive. Stopping trace.'%t 
                break
            elif t < T_MIN:
                #print 'Warning: t = %f is too negative. Stopping trace.'%t 
                break

    # Sort points along the MAP curve
    map_curve.sort()
    map_curve.t_start = t_start

    # Return the MAP curve to the user
    return map_curve

# Sample plausible densities from the posterior: 
def sampling( points, R, Delta, N, G, coeffs_infty, kernel_basis, num_samples, \
              num_steps_per_sample, num_thermalization_steps, fix_t_at_t_star):
    start_time = time.clock()
    Q_samples = np.zeros([G,num_samples])
    sample_index = 0

    # Read t, Q, log_E from points
    t = sp.array([p.t for p in points])
    Q = sp.array([p.Q for p in points])
    log_E = sp.array([p.log_E for p in points])
    num_t = len(t)
    print
    print 't values: MaxEnt <--> R'
    print t, num_t
    print
    # Generate a "histogram" of t according to their relative probability
    log_E = log_E - log_E.max()
    prob_t = sp.exp(log_E)
    prob_t = prob_t/sp.sum(prob_t)
    num_indices = num_t
    sampled_indices = list(np.random.choice( num_indices, size=num_samples, \
                                             replace=True, p=prob_t))
    hist_t = [sampled_indices.count(c) for c in range(num_indices)]
    print 'Histogram of t:'
    print hist_t, sp.sum(hist_t)
    print 
    # If want to fix t at t_star
    if fix_t_at_t_star:
        imax = log_E.argmax()
        hist_t = np.ndarray(num_t)
        hist_t[:] = 0
        hist_t[imax] = num_samples
        print 'Histogram of t: Fix t at t_star'
        print hist_t, sp.sum(hist_t)
        print 
    # Output prob_t vs t
    prob_t_vs_t = np.ndarray((2,num_t))
    prob_t_vs_t[0,:] = t[:]
    prob_t_vs_t[0,0] = t[1] - 10
    prob_t_vs_t[0,num_t-1] = t[num_t-2] + 10
    prob_t_vs_t[1,:] = prob_t[:]

    # Traverse t distribution
    print 'Traverse t distribution:'
    for i in range(num_t):
        if (hist_t[i] > 0):
            print i, t[i], hist_t[i]
            if (t[i] == -sp.Inf): # t is -infinity
                sub_num_samples = hist_t[i]
                # Find eigen-modes of the hessian for later use
                H = maxent.hessian_per_datum_from_coeffs( coeffs_infty, R, \
                                                          kernel_basis, phi0=False, \
                                                          regularized=True) * G
                H = sp.mat(H)
                D = np.linalg.eig(H)
                # Kernel dimension and number of modes
                kernel_dim = kernel_basis.shape[1]
                num_modes = kernel_dim
                
                # Initializing
                step_vector = np.zeros(kernel_dim)
                coeffs_current = coeffs_infty
                S_current = maxent.action_per_datum_from_coeffs( coeffs_current, R, \
                                                                 kernel_basis, phi0=False, \
                                                                 regularized=True) * G

                # Thermalizing and the sampling
                for k in range( 1, num_thermalization_steps+ \
                                   int(sub_num_samples*num_steps_per_sample)+1):
                    g = np.random.randint(0,num_modes)
                    e_val = D[0][g]
                    e_vec = D[1][:,g].T
                    step_length = np.random.normal(0,1.0/(np.sqrt(e_val)))
                    step_vector = step_length * e_vec
                    coeffs_mat = coeffs_current + step_vector # np.mat(1,kernel_dim)
                    coeffs_new = np.ndarray(kernel_dim)
                    coeffs_new[:] = coeffs_mat[0,:] # np.ndarray(kernel_dim,)
                    S_new = maxent.action_per_datum_from_coeffs( coeffs_new, R, \
                                                                 kernel_basis, phi0=False, \
                                                                 regularized=True) * G
                    if np.random.uniform(0,1) < np.exp((S_current-S_new)*N/G):
                        coeffs_current = coeffs_new
                        S_current = S_new
                    if (k > num_thermalization_steps and k%num_steps_per_sample == 0):
                        phi = maxent.coeffs_to_field(coeffs_current, kernel_basis)
                        Q_samples[:,sample_index] = utils.field_to_prob(phi)
                        sample_index += 1
                        
            else: # t is finite
                sub_num_samples = hist_t[i]
                Q_t = np.ndarray(G)
                Q_t[:] = Q[i,:]
                phi_t = utils.prob_to_field(Q_t)

                # Find eigen-modes of the hessian for later use
                H = hessian(phi_t, R, Delta, t[i], N, regularized=True)
                H_prime = H.todense()
                D = np.linalg.eig(H_prime)
                # Number of eigen-modes
                num_modes = G

                # Initializing
                step_vector = np.zeros(G)
                phi_current = phi_t
                S_current = action( phi_current, R, Delta, t[i], N, phi_in_kernel=False, \
                                    regularized=True)

                # Thermalizing and then sampling
                for k in range( 1, num_thermalization_steps+ \
                                   int(sub_num_samples*num_steps_per_sample)+1):
                    g = np.random.randint(0,num_modes)
                    e_val = D[0][g]
                    e_vec = D[1][:,g].T
                    step_length = np.random.normal(0,1.0/(np.sqrt(e_val)))
                    step_vector = step_length * e_vec
                    phi_mat = phi_current + step_vector # np.mat(1,G)
                    phi_new = np.ndarray(G) 
                    phi_new[:] = phi_mat[0,:] # np.ndarray(G,)
                    S_new = action( phi_new, R, Delta, t[i], N, phi_in_kernel=False, \
                                    regularized=True)
                    if np.random.uniform(0,1) < np.exp((S_current-S_new)*N/G):
                        phi_current = phi_new
                        S_current = S_new
                    if (k > num_thermalization_steps and k%num_steps_per_sample == 0):
                        Q_samples[:,sample_index] = utils.field_to_prob(phi_current)
                        sample_index += 1
    
    end_time = time.clock()
    posterior_sample_compute_time = end_time - start_time

    return prob_t_vs_t, hist_t, Q_samples, posterior_sample_compute_time

### Core DEFT algorithm ###

# The core algorithm of DEFT, used for both 1D and 2D density esitmation
def run(counts_array, Delta, resolution=3.14E-2, tollerance=1E-3, details=False, \
    errorbars=False, Laplace=True, num_samples=0, t_start=0.0, print_t=False, \
    num_steps_per_sample='default', num_thermalization_steps='default', \
    fix_t_at_t_star=False):
    """
    The core algorithm of DEFT, used for both 1D and 2D density estmation.

    Args:
        counts_array (numpy.ndarray): 
            A scipy array of counts. All counts must be nonnegative.

        Delta (Smoothness_operator instance): 
            An operator providing the definition of 'smoothness' used by DEFT
    """

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure the smoothness_operator has the right shape
    assert(G == len(counts_array))

    # Make sure histogram is nonnegative
    assert(all(counts_array >= 0))

    # Make sure that enough elements of counts_array contain data
    assert(sum(counts_array >= 0) > kernel_dim)

    # Get number of data points and normalized histogram
    N = sum(counts_array)

    # Get normalied histogram
    R = 1.0*counts_array/N
    
    # Compute the MAP curve
    start_time = time.clock()
    map_curve = compute_map_curve( N, R, Delta, \
                                   resolution=resolution, \
                                   tollerance=tollerance, \
                                   t_start=t_start, \
                                   print_t=print_t, \
                                   Laplace=Laplace)
    end_time = time.clock()
    map_curve_compute_time = end_time-start_time
    if print_t:
        print 'MAP curve computation took %.2f sec'%(map_curve_compute_time)

    # Identify the optimal density estimate
    points = map_curve.points
    log_Es = sp.array([p.log_E for p in points])
    log_E_max = log_Es.max()
    ibest = log_Es.argmax()
    star = points[ibest]
    Q_star = np.copy(star.Q)
    t_star = star.t
    phi_star = np.copy(star.phi)
    map_curve.i_star = ibest
    coeffs_infty = map_curve.coeffs_infty
    kernel_basis = map_curve.kernel_basis

    # Sample plausible densities from the posterior:
    if (num_samples > 0):
        prob_t_vs_t, hist_t, Q_samples, posterior_sample_compute_time = \
        sampling( points, R, Delta, N, G, coeffs_infty, kernel_basis, num_samples, \
                  num_steps_per_sample, num_thermalization_steps, fix_t_at_t_star)
        
    #
    # Package results
    #

    # Create container
    results = Results()
    
    # Fill in info that's guareneed to be there
    results.Q_star = Q_star
    results.R = R
    results.map_curve = map_curve
    results.map_curve_compute_time = map_curve_compute_time
    results.G = G
    results.N = N
    results.t_star = t_star
    results.i_star = ibest
    results.counts = counts_array
    results.resolution = resolution
    results.tollerance = tollerance
    #results.Delta = Delta
    results.errorbars = errorbars
    results.num_samples = num_samples

    # Include errorbar info if this was computed
    if errorbars:
        results.Q_ub = Q_ub
        results.Q_lb = Q_lb
        results.errorbar_compute_time = errorbar_compute_time

    # Include posterior sampling info if any sampling was performed
    if num_samples > 0:
        results.prob_t_vs_t = prob_t_vs_t
        results.hist_t = hist_t
        results.Q_samples = Q_samples
        results.posterior_sample_compute_time = posterior_sample_compute_time

    # Return density estimate along with histogram on which it is based
    return results

