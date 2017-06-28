### This is a place to store things that are not used at this moment. ###

    # Compute errorbars if requested
    if errorbars:   
        start_time = time.clock()

        # Get list of map_curve points that with evidence ratio 
        # of at least 1% the maximum
        log_E_threshold = log_E_max + np.log(0.001)

        # Get points that satisfy threshold
        points_at = [p for p in points if p.log_E > log_E_threshold]

        #print '\n'.join(['%f\t%f'%(p.t,p.log_E) for p in points])

        # Get weights at each ell
        log_Es_at = np.array([p.log_E for p in points_at])
        log_Es_at -= log_Es_at.max()
        weight_ell = np.mat(np.exp(log_Es_at))
        
        # Get systematic variance due to changes in Q_ell at each ell
        dQ_sq_sys_ell = np.mat([(p.Q-Q_star)**2 for p in points_at])

        # Get random variance about Q_ell at each ell
        dQ_sq_rand_ell = np.mat([get_dQ_sq(p.phi, R, Delta, p.t, N) 
            for p in points_at])

        #print dQ_sq_rand_ell

        # Sum systematic and random components to variance
        dQ_sq_ell = dQ_sq_sys_ell + dQ_sq_rand_ell

        #print weight_ell.shape
        #print dQ_sq_ell.shape

        # Compute weighted averaged to get final dQ_sq
        dQ_sq_mat = weight_ell*dQ_sq_ell/sp.sum(sp.array(weight_ell))
        
        # Convert from matrix to array
        dQ_sq = sp.array(dQ_sq_mat).ravel()
        
        try:
            assert(all(np.isfinite(dQ_sq)))
        except:
            print [p.log_E for p in points_at]
            print weight_ell
            print dQ_sq_sys_ell
            print dQ_sq_rand_ell
            raise

        # Compute interval
        Q_ub = Q_star + np.sqrt(dQ_sq)
        Q_lb = Q_star - np.sqrt(dQ_sq)

        # Compute time to get errorbars
        end_time = time.clock()
        errorbar_compute_time = end_time - start_time

    # Sample plausible densities from the posterior
    Q_samples = sp.zeros([0,0])
    if num_samples > 0:
        start_time = time.clock()

        #print 't_star == ' + str(t_star)

        # Get list of map_curve points that with evidence ratio 
        # of at least 1% the maximum
        log_E_threshold = log_E_max + np.log(0.001)

        # Get points that satisfy threshold
        points_at = [p for p in points if p.log_E > log_E_threshold]

        # Get weights at each ell
        weights = np.array([np.exp(p.log_E) for p in points_at])

        # Compute eigenvectors of the Hessian
        # If t is finite, this is straight-forward
        if t_star > -np.Inf:
            h_star = hessian(phi_star, R, Delta, t_star, N, regularized=False)
            lambdas_unordered, psis_unordered = eigh(h_star.todense())
            ordered_indices = np.argsort(lambdas_unordered)
            psis = psis_unordered[:,ordered_indices]

        # If t is infinite but kernel is non-degenerate
        elif Delta._kernel_dim == 1:
            psis = Delta._eigenbasis

        # If t is infinite and kernel is degenerate and needs to be
        # diagonalized with respect to diag(Q_star)
        else:
            psis_ker = Delta._kernel_basis
            kd = Delta._kernel_dim
            h_ker = sp.zeros([kd, kd])
            psis = sp.zeros([G,G])
            for i in range(kd):
                for j in range(kd):
                    psi_i = sp.array(psis_ker[:,i])
                    psi_j = sp.array(psis_ker[:,j])
                    h_ker[i,j] = sp.sum(np.conj(psi_i)*psi_j*Q_star)
            _ , cs = eigh(h_ker)
            rhos = sp.mat(cs).T*psis_ker.T
            psis[:,:kd] = rhos.T
            psis[:,kd:] = Delta._eigenbasis[:,kd:]
            
        # Figure out how many samples to draw for each ell value
        candidate_ell_indices = range(len(points_at))
        candidate_ell_probs = weights/sp.sum(weights)
        ell_indices = np.random.choice(candidate_ell_indices, 
            size=num_samples, p=candidate_ell_probs)
        unique_ell_indices, ell_index_counts = np.unique(ell_indices, return_counts=True)

        # Draw samples at each lenghtscale
        Q_samples = sp.zeros([G,num_samples])
        num_samples_obtained = 0
        for k in range(len(unique_ell_indices)):
            ell_index = unique_ell_indices[k]
            num_samples_at_ell = ell_index_counts[k]

            p = points_at[ell_index]

            # If t is finite, figure out how many psis to use
            if p.t > -np.Inf:
                # Get hessian
                #H = (1.*N/G)*hessian(p.phi, R, Delta, p.t)
                H = hessian(p.phi, R, Delta, p.t, N, regularized=False)

                # Compute inverse variances below threshold
                inv_vars = []
                for i in range(int(G)):
                    psi = psis[:,i]
                    psi_col = sp.mat(psi[:,None])
                    inv_var = (np.conj(psi_col.T)*H*psi_col)[0,0]
                    if i==0:
                        inv_vars.append(inv_var)
                    elif inv_var < (1.0E10)*min(inv_vars):
                        inv_vars.append(inv_var)
                    else:
                        break;
                assert all(np.isreal(inv_vars))
                psis_use = psis[:,:len(inv_vars)]

            # If t is finite, only use psis in kernel
            else:
                #H = 1.*N*spdiags(p.Q,0,G,G)
                H = 1.*G*spdiags(p.Q,0,G,G)
                kd = Delta._kernel_dim
                psis_use = psis[:,:kd]
                inv_vars = sp.zeros(kd)
                for i in range(kd):
                    psi_i = sp.mat(psis_use[:,i]).T
                    inv_var = (np.conj(psi_i.T)*H*psi_i)[0,0]
                    assert np.isreal(inv_var)
                    inv_vars[i] = inv_var
                
            # Make sure all inverse variances are greater than zero
            assert all(np.array(inv_vars) > 0)

            # Now draw samples at this ell!
            psis_use_mat = sp.mat(sp.array(psis_use))
            inv_vars = sp.array(inv_vars)
            num_psis_use = psis_use_mat.shape[1]
            
            # Perform initial sampling at this ell
            # Sample 10x more phis than needed if doing posterior pruning
            M = 10*num_samples_at_ell
            #M = num_samples_at_ell
            phi_samps = sp.zeros([G,M])
            sample_actions = sp.zeros(M)
            for m in range(M):
                
                # Draw random numbers for dphi coefficients
                r = sp.randn(num_psis_use)

                # Compute action used for sampling
                S_samp = np.sum(r**2)/2.0 # Action for specific sample

                # Construct sampled phi
                sigmas = 1./np.sqrt((1.*N/G)*inv_vars)
                a = sp.mat(r*sigmas)
                dphi = sp.array(a*psis_use_mat.T).ravel()
                phi = p.phi + dphi
                phi_samps[:,m] = phi
                
                # Compute true action for phi_samp
                phi_in_kernel = (p.t == -np.Inf)

                # USE THIS IF YOU DON'T WANT TO DO POSTERIOR PRUNING
                # RIGHT NOW I DON'T THINK THIS SHOULD BE DONE
                # THIS LACK OR PRUNING CREATES FLIPPY TAILS ON THE POSTERIOR
                # SAMPLES, BUT THIS GENUINELY REFLECTS THE HESSIAN I THINK
                if False:
                    sample_actions[m] = 0
                else:
                    S = (1.*N/G)*action(phi, R, Delta, p.t, N, phi_in_kernel=phi_in_kernel, regularized=False)
                    sample_actions[m] = S-S_samp

            # Now compute weights. Have to make bring actions into a 
            # sensible range first
            sample_actions -= sample_actions.min()
            

            # Note: sometimes all samples except one have nonzero weight
            # The TINY_FLOAT32 here regularizes these weights so that
            # the inability to sample well doesn't crash the program
            sample_weights = sp.exp(-sample_actions) + utils.TINY_FLOAT32

            # Choose a set of samples. Do WITHOUT replacement.
            try:
                sample_probs = sample_weights/np.sum(sample_weights)
                sample_indices = sp.random.choice(M, size=num_samples_at_ell, replace=False, p=sample_probs)
            except:
                print sample_weights
                print sample_probs
                print num_samples_at_ell
                raise

            #print p.t
            #print sample_weights
            #print np.sort(sample_probs)[::-1]
            for n in range(num_samples_at_ell):
                index = sample_indices[n]
                #print sample_weights[index]
                phi = phi_samps[:,index]
                m = num_samples_obtained + n
                Q_samples[:,m] = utils.field_to_prob(phi)

            num_samples_obtained += num_samples_at_ell

        # Randomize order of samples
        indices = np.arange(Q_samples.shape[1])
        np.random.shuffle(indices)
        Q_samples = Q_samples[:,indices]
        end_time = time.clock()
        posterior_sample_compute_time = end_time-start_time

#-----------------------------------------------------------------------
            
    # Feynman diagrams of order N^{-2}
    S = np.zeros(16) # 1 zero + 15 diagrams
    
    # 1-vertex
    for i in range(G):
        Vi = V[i]
        S[1] += Vi * P[i,i]**3
    S[1] = (-1)*S[1]/48
    
    # 2-vertex
    for i in range(G):
        for j in range(G):
            Vij = V[i]*V[j]
            S[2] += Vij * P[i,i]*P[i,j]**2*P[j,j]
            S[3] += Vij * P[i,j]**4
            S[4] += Vij * P[i,i]*P[i,j]*P[j,j]**2
            S[5] += Vij * P[i,j]**3*P[j,j]
    S[2] = S[2]/16
    S[3] = S[3]/48
    S[4] = S[4]/16
    S[5] = S[5]/12

    # 3-vertex
    for i in range(G):
        for j in range(G):
            for k in range(G):
                Vijk = V[i]*V[j]*V[k]
                S[6] += Vijk * P[i,i]*P[i,j]*P[j,k]**2*P[k,k]
                S[7] += Vijk * P[i,i]*P[i,j]*P[j,j]*P[i,k]*P[k,k]
                S[8] += Vijk * P[i,k]**3*P[i,j]*P[j,j]
                S[9] += Vijk * P[i,i]*P[i,j]*P[i,k]*P[j,k]**2
                S[10] += Vijk * P[i,j]**2*P[i,k]**2*P[j,k]
    S[6] = (-1)*S[6]/8
    S[7] = (-1)*S[7]/16
    S[8] = (-1)*S[8]/12
    S[9] = (-1)*S[9]/8
    S[10] = (-1)*S[10]/8

    # 4-vertex
    for i in range(G):
        for j in range(G):
            for k in range(G):
                for l in range(G):
                    Vijkl = V[i]*V[j]*V[k]*V[l]
                    S[11] += Vijkl * P[i,j]*P[i,k]**2*P[j,l]**2*P[k,l]
                    S[12] += Vijkl * P[i,i]*P[i,j]*P[j,k]**2*P[k,l]*P[l,l]
                    S[13] += Vijkl * P[i,j]*P[j,j]*P[i,k]*P[k,k]*P[i,l]*P[l,l]
                    S[14] += Vijkl * P[i,l]*P[l,l]*P[i,j]*P[i,k]*P[j,k]**2
                    S[15] += Vijkl * P[i,j]*P[i,k]*P[i,l]*P[j,k]*P[k,l]*P[l,j]
    S[11] = S[11]/8
    S[12] = S[12]/16
    S[13] = S[13]/48
    S[14] = S[14]/8
    S[15] = S[15]/24
    

