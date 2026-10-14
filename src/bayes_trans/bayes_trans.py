#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  bayes_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2025

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from tensorflow_probability.substrates import jax as tfp
tfpd = tfp.distributions

def d2t(deltas):
    L = len(deltas)+1
    itemps = np.ones(L)
    for l in range(L-1):
        itemps[l+1] = itemps[l] * np.exp(-np.exp(deltas[l]))
    temps = 1/itemps
    return(temps)

########################################
## ETA NLD ## 
# Negative log density of eta (up to eta-free constants), batched over rows of `etas`.
# rb selects what is integrated out; comp_prior selects the prior on the complement of z=B beta.
def _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, rb, P, K, omega_prior, comp_prior):
    E0 = np.concatenate([np.eye(P)]+[np.zeros([P,P]) for _ in range(K-1)],axis=1)
    Bval = get_B(grams_all, etas, Ns, tau2)

    ####
    # Density is of the form C_rB * exp(-n_rB) times C_kB * exp(-n_kb)
    ####

    if rb=='betasig' and comp_prior=='shear':
        ## Shear prior, beta and sigma2 integrated out.
        # Work in sheared coordinates gamma = (z, beta_S), z = beta_0 - T beta_S (unit Jacobian). There the
        # prior is diagonal, A := blockdiag(Omega^{-1}, G_k + lambda_p I), and the posterior precision is
        #   Q = A + C^T G_0 C,   C = [I_P, T]   (P x KP),
        # a block-diagonal-plus-rank-P matrix, so everything reduces to PxP algebra (matrix det lemma / Woodbury).
        nE = etas.shape[0]
        ld = lambda X: jnp.linalg.slogdet(X)[1]
        G0 = grams_all[0]
        A_src = grams_all[1:] + vw['lambda_p']*jnp.eye(P)[None,:,:]      # [K-1,P,P], eta-free
        A_src_inv = jnp.linalg.inv(A_src)
        Tm = -Bval[:,:,P:]                                                # [nE,P,(K-1)P]
        Tk = Tm.reshape([nE,P,K-1,P]).transpose([0,2,1,3])                # [nE,K-1,P,P]
        TkT = jnp.matrix_transpose(Tk)
        # V = C A^{-1} C^T = Omega + sum_k T_k A_k^{-1} T_k^T
        V = jnp.diag(vw['omega'])[None,:,:] + jnp.sum(Tk @ A_src_inv[None] @ TkT, axis=1)
        IpGV = jnp.eye(P)[None,:,:] + G0[None,:,:] @ V                    # [nE,P,P]

        pri_prec_logdet = (K-1)*P*jnp.log(vw['lambda_p']) - jnp.sum(jnp.log(vw['omega']))
        post_prec_logdet = jnp.sum(ld(A_src)) - jnp.sum(jnp.log(vw['omega'])) + ld(IpGV)
        logdet_term = 0.5*(pri_prec_logdet - post_prec_logdet)

        # Quadratic term: b^T Q^{-1} b with b = C^T x_0 + [0; x_k], x_k = X_k^T y_k, via
        # (A + C^T W C)^{-1} = A^{-1} - A^{-1} C^T (I + W V)^{-1} W C A^{-1}  (no G_0 inverse needed).
        Xty = (Xa.T @ ya).reshape([K,P])
        x0 = Xty[0]
        bk = (TkT @ x0[None,None,:,None])[:,:,:,0] + Xty[1:][None,:,:]   # [nE,K-1,P]
        Aibk = (A_src_inv[None] @ bk[:,:,:,None])[:,:,:,0]               # [nE,K-1,P]
        bAb = jnp.sum(vw['omega']*jnp.square(x0)) + jnp.sum(bk*Aibk, axis=(1,2))
        c = vw['omega']*x0[None,:] + jnp.sum((Tk @ Aibk[:,:,:,None])[:,:,:,0], axis=1)  # [nE,P]
        G0c = (G0[None] @ c[:,:,None])
        quad = bAb - (c[:,None,:] @ jnp.linalg.solve(IpGV, G0c))[:,0,0]
        n_term = -Xa.shape[0]/2*jnp.log(jnp.sum(jnp.square(ya)) - quad)

        ll = logdet_term + n_term
        nll = -ll

        # Complement component is just beta_S under the shear prior (zero-padded on the target block).
        resids = jnp.tile(jnp.concatenate([jnp.zeros(P), vw['beta'][1:,:].flatten()]),[nE,1])
    elif rb=='betasig':
        svd = jnp.linalg.svd(Bval,full_matrices=False)
        Ub = svd[2]

        ### Slow but explicit code.
        #pri_prec1 = vw['lambda_p']*(jnp.eye(K*P)-jnp.matrix_transpose(Ub)@Ub)
        #pri_prec2 = jnp.matrix_transpose(Bval)@jnp.diag(1/vw['omega'])@Bval
        #pri_prec = pri_prec1 + pri_prec2
        #pri_prec_logdet = jnp.linalg.slogdet(pri_prec)[1]
        ### Slow but explicit code.

        # TODO: DRY
        ## fast but implicit code.
        disp = -0.5*jnp.sum(jnp.log(vw['omega'])) 
        #Bdetterm = jnp.linalg.slogdet(sob @ jnp.matrix_transpose(sob))[1]
        detBBT = jnp.linalg.slogdet(Bval @ jnp.matrix_transpose(Bval))[1] 
        #C_rB = 0.5*detBBT + disp
        C_rB = 0.5*detBBT + disp
        # C_rb gives the log normalizing constant for the distribution of beta in the range of B.
        C_kB = (K-1)*P/2 * jnp.repeat(jnp.log(vw['lambda_p']), etas.shape[0])
        # C_rb gives the log normalizing constant for the distribution of beta in the kernel of B.
        logC = C_rB + C_kB
        pri_prec_logdet = 2*logC
        ## fast but implicit code.
        # TODO: DRY

        ### Slow but explicit code.
        #GG = block_diag(*[grams_all[k,:,:] for k in range(K)])
        #post_prec = GG+pri_prec
        #post_prec_logdet = jnp.linalg.slogdet(post_prec)[1]
        ### Slow but explicit code.

        ## fast but implicit code.
        # Just matrix det lemma.
        grams_aug_all = grams_all + vw['lambda_p']*jnp.eye(P)[None,:,:]
        Btens = Bval.transpose([0,2,1]).reshape([etas.shape[0],K,P,P])
        VtAU = Bval @ (jnp.linalg.inv(grams_aug_all[None,:,:,:]) @ Btens).reshape([etas.shape[0],K*P,P])
        W = jnp.diag(1/vw['omega'])[None,:,:] - vw['lambda_p']*jnp.linalg.inv(Bval@jnp.matrix_transpose(Bval))
        ld = lambda X: jnp.linalg.slogdet(X)[1]
        ldA = jnp.sum(ld(grams_aug_all))
        DELTA = jnp.linalg.inv(W)+VtAU
        post_prec_logdet = ld(DELTA) + ld(W) + ldA[None]
        #VtAU = Bval @ block_diag(*grams_all) @ jnp.matrix_transpose(Bval)
        ## fast but implicit code.

        logdet_term = 0.5*(pri_prec_logdet - post_prec_logdet)

        ### Slow but explicit code.
        #yhat = (Xa[None,:,:] @ jnp.linalg.solve(post_prec, Xa.T @ ya)[:,:,None])[:,:,0]
        ### Slow but explicit code.

        ## fast but implicit code.
        Xty = Xa.T @ ya
        AiXty = jnp.linalg.solve(grams_aug_all, Xty.reshape([K,P,1]))
        AiXty = AiXty.reshape([K*P,1])
        #Ai = jnp.linalg.inv(block_diag(*grams_all) + vw['lambda_p']*jnp.eye(K*P))
        #AiXty = Ai @ Xty
        mm = jnp.matrix_transpose(Bval)@ (jnp.linalg.solve(DELTA, (Bval @ AiXty[None,:,:])))
        Aimm = jnp.linalg.solve(grams_aug_all[None,:], mm.reshape([etas.shape[0],K,P,1]))
        #jnp.matrix_transpose(Bval) @ (DELTA @ (Bval @ (Ai @ Xty))[:,:,None])
        #Aimm = Ai[None,:,:] @ mm
        #diff = AiXty[None,:,None] - Aimm
        #yhat = Xa[None,:,:] @ diff
        Aimm = Aimm.reshape([etas.shape[0],K*P,1])
        yhat = (Xa[None,:,:] @ (AiXty[None,:,:] - Aimm))[:,:,0]
        ## fast but implicit code.

        n_term = -Xa.shape[0]/2*jnp.log(jnp.sum(ya[None,:]*(ya[None,:]-yhat),axis=1))

        ll = logdet_term + n_term
        nll = -ll

        v = jnp.tile(vw['beta'].flatten(),[etas.shape[0],1])[:,None,:]
        sol = jax.vmap(lambda X: jnp.linalg.lstsq(X[:-1,:].T,X[-1,:])[0])(jnp.concatenate([Bval,v],axis=1))
        resids = v[:,0,:] - (jnp.matrix_transpose(Bval) @ sol[:,:,None])[:,:,0]
    else:
        ####
        ## Form the (log of the) term in front of the exponential, involving determinants.
        if rb=='omega':
            if omega_prior=='exp':
                disp = P*(jnp.log(vw['lambda_t']) - 0.5*jnp.log(vw['sigma2']) - jnp.log(2))
            elif omega_prior=='ig':
                disp = P*(jnp.log(vw['lambda_t']) - 0.5*jnp.log(vw['sigma2']) - jnp.log(np.pi))
            else:
                raise Exception("Oh noes!")
        else:
            disp = -0.5*jnp.sum(jnp.log(vw['omega'])) - P/2.*jnp.log(2*np.pi*vw['sigma2'])

        if comp_prior=='shear':
            # Unit-Jacobian shear: no det(B B^T) term.
            C_rB = disp + jnp.zeros(etas.shape[0])
        else:
            detBBT = jnp.linalg.slogdet(Bval @ jnp.matrix_transpose(Bval))[1]
            C_rB = 0.5*detBBT + disp
        C_kB = (K-1)*P/2 * (jnp.repeat(jnp.log(vw['lambda_p']), etas.shape[0]) - jnp.log(2*np.pi*vw['sigma2']))
        logC = C_rB + C_kB
        ####

        ####
        ## Form the (negation of the log of the) term in the exponential, involving norms.
        v = jnp.tile(vw['beta'].flatten(),[etas.shape[0],1])[:,None,:]
        if comp_prior=='shear':
            # Complement component is beta_S itself; pad with zeros on the target block so that
            # sum(v*resids) below gives ||beta_S||^2.
            resids = jnp.tile(jnp.concatenate([jnp.zeros(P), vw['beta'][1:,:].flatten()]),[etas.shape[0],1])
        else:
            sol = jax.vmap(lambda X: jnp.linalg.lstsq(X[:-1,:].T,X[-1,:])[0])(jnp.concatenate([Bval,v],axis=1))
            resids = v[:,0,:] - (jnp.matrix_transpose(Bval) @ sol[:,:,None])[:,:,0]
        neg_n_kB = 0.5* vw['lambda_p'] * jnp.sum(v[:,0,:]*resids,axis=1) / vw['sigma2']
        if rb=='omega':
            z = Bval @ jnp.matrix_transpose(v)
            z = z[:,:,0]
            if omega_prior=='exp':
                neg_n_rB = vw['lambda_t']/jnp.sqrt(vw['sigma2'])*jnp.sum(jnp.abs(z),axis=1) # Second axis is singleton.
            elif omega_prior=='ig':
                neg_n_rB = jnp.sum(jnp.log1p(jnp.square(vw['lambda_t'])/vw['sigma2']*jnp.square(z)),axis=1)
            else:
                raise Exception("Unknown omega_prior.")
        else:
            neg_n_rB = jnp.matrix_transpose(Bval) @ (jnp.diag(1/vw['omega'])[None,:,:] @ (Bval@jnp.matrix_transpose(v)))
            neg_n_rB = neg_n_rB[:,:,0]
            neg_n_rB = 0.5 * jnp.sum(v[:,0,:]*neg_n_rB,axis=1) / vw['sigma2']
        neg_norm = neg_n_kB + neg_n_rB
        # neg_n_kB gives the opposite of the term in the exponent for the distribution of the part of beta in the kernel of B. 
        # neg_n_rB same but for the part in the range.
        ####

        nll = -logC + neg_norm

    prior_contrib = -jnp.sum(tfpd.Bernoulli(probs=vw['rho']).log_prob(etas),axis=1)
    nlog_probs = nll + prior_contrib
    return nlog_probs, resids

def bayes_trans(Xs, ys, iters = 10000, fixed = {}, plotprefix = None, desired_ar = 0.574, desired_sr = 0.2, gamma_exp = 1., sigma_min = 1e-4, verbose = True, debug = False, omega_prior = 'exp', rb = 'betasig', L = 5, true_vals = None, err_Cs = None, comp_prior = 'shear'):
    """
    rb: which parameters are integrated out (Rao-Blackwellized) when sampling eta.
        One of 'betasig' (default), 'none', 'omega', 'alt'.
    comp_prior: prior on the (K-1)P directions complementary to the transfer direction z = B beta.
        'shear' is the default; 'orth' reproduces versions <= 0.1.0.
        'orth'  (legacy): N(0, sigma2/lambda_p) on the orthogonal complement of row(B), i.e. prior
                precision lambda_p*(I - P_B) + B^T Omega^{-1} B. Normaliser carries sqrt(det(B B^T)).
        'shear': N(0, sigma2/lambda_p) on the source coefficients beta_S themselves, i.e. prior
                precision lambda_p*D + B^T Omega^{-1} B with D = diag(0_P, I_{(K-1)P}). The map
                (beta_0, beta_S) -> (z, beta_S) is a unit-Jacobian shear, so the normaliser is eta-free.
    """

    Ns = jnp.array([X.shape[0] for X in Xs])
    Ps = [X.shape[1] for X in Xs]
    assert np.all([P==Ps[0] for P in Ps])
    P = Ps[0]
    K = len(Ns)

    docorr = err_Cs is not None

    burnin = np.minimum(5000,iters//5)
    eta_freeze = burnin//2

    assert rb in ['none','omega','betasig','alt']
    assert comp_prior in ['orth','shear']
    if verbose:
        print(f"bayes_trans settings: comp_prior={comp_prior}, rb={rb}, omega_prior={omega_prior}, L={L}, iters={iters}, K={K}, P={P}, fixed={list(fixed.keys())}")

    ## Default init vals; this is also how the program keeps track of what variables there are.
    di = {}
    di['beta'] = np.random.normal(size=[K,P])
    di['eta'] = jnp.zeros([K-1])
    di['sigma2'] = jnp.array(1.)
    di['lambda_t'] = jnp.array(1.)
    di['lambda_p'] = jnp.array(1.)
    di['rho'] = jnp.array(0.5)
    di['omega'] = jnp.ones(([P]))
    di['tau2'] = jnp.array(1.)
    if docorr:
        di['psi'] = jnp.array(1.)

    etas = jnp.tile(di['eta'][None,:], [L,1])

    tau2_prop_scale = 1.
    tau2_accepts = 0.
    tau2_rejects = 0.

    if docorr: #TODO: unified scale initializtion?
        psi_prop_scale = 1.
        psi_accepts = 0.
        psi_rejects = 0.
    
    who = np.zeros(iters)
    didswitch = np.zeros(iters)
    etas_accepts = np.zeros([iters,L])

    tracking = {}
    for v in di:
        tracking[v] = np.nan*np.zeros([iters]+list(di[v].shape))
        tracking[v][0] = fixed[v] if v in fixed else di[v]


    #TODO: surely we can do this with more DRY.
    eta_nld_omega = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, 'omega', P, K, omega_prior, comp_prior))
    eta_nld_betasig = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, 'betasig', P, K, omega_prior, comp_prior))
    eta_nld_none = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, 'none', P, K, omega_prior, comp_prior))
    ######################################################################################################

    vw = dict([(v,jnp.array(tracking[v][0])) for v in tracking])

    # Initial partemp params
    init_max_T = 1000.
    deltas = np.zeros(L-1)
    delta_tracking = np.zeros([iters,L-1])

    Bset = False
    for i in tqdm(range(1,iters), leave = False, disable = not verbose):
        if rb=='betasig' or (rb=='alt' and i % 3 == 0):
            eta_nld = eta_nld_betasig
        elif rb=='omega' or (rb=='alt' and i % 3 == 1):
            eta_nld = eta_nld_omega
        elif rb=='none' or (rb=='alt' and i % 3 == 2):
            eta_nld = eta_nld_none
        else:
            raise Exception("Unknown rb!")

        ## Whiten
        ### X/y STUFF
        LXs = [Xs[k] for k in range(len(Xs))]
        Lys = [ys[k] for k in range(len(ys))]
        if docorr:
            def get_Lk(psi, k):
                err_mat = np.linalg.inv(psi*err_Cs[k]+np.eye(Ns[k]))
                Lk = np.linalg.cholesky(err_mat)
                return Lk

            print("Whitening...")
            for k in range(len(Xs)):
                Lk = get_Lk(vw['psi'], k)
                LXs[k] = Lk.T@Xs[k]
                Lys[k] = Lk.T @ ys[k]

        # TODO: no recompute.
        beta_hat = np.concatenate([np.linalg.lstsq(LXs[k], Lys[k])[0] for k in range(K)])
        bha = np.stack([np.linalg.lstsq(LXs[k], Lys[k])[0] for k in range(K)])

        ## Precompute X quantities.
        # TODO: can be improved by using QR or SVD on X.
        grams_all = jnp.stack([LXs[k].T @ LXs[k] for k in range(K)])
        grams_chols = jnp.linalg.cholesky(grams_all)
        grams_invchols = jnp.linalg.inv(grams_chols)
        grams_inv_all = jnp.linalg.inv(grams_all)
        Xty = jnp.stack([LXs[k].T @ Lys[k] for k in range(K)])[:,:,np.newaxis]

        # TODO: Can we avoid forming this?
        Xa = block_diag(*LXs)
        Na = np.sum(Ns)
        ya = np.concatenate(Lys)
        ### X/y STUFF

        # On first iteration, need to generate transfer matrix here. 
        # TODO: Can this be done nicer?
        if not Bset:
            Bset = True
            B = get_B(grams_all, vw['eta'][None,:], Ns, vw['tau2'])[0,:,:]
            z_samp = B @ vw['beta'].flatten()

        ### Also X/y stuff.
        grams_aug_all = jnp.stack([grams_all[k,:,:] + vw['lambda_p']*jnp.eye(P) for k in range(K)])
        #TODO: can be updated more efficient from XTX chols.
        grams_aug_chols = jnp.linalg.cholesky(grams_aug_all)
        grams_aug_invchols = jnp.linalg.inv(grams_aug_chols) # NOTE: this really is the inverse of the chol; it is not the chol of the inverse (its the transpose of that!).
        grams_aug_inv_all = jnp.linalg.inv(grams_aug_all)
        bha_aug = grams_aug_inv_all @ Xty
        ### Also X/y stuff.

        if 'omega' in fixed:
            pass
        else:
            if omega_prior=='exp':
                nu_ig = np.sqrt(vw['sigma2']*np.square(vw['lambda_t']/z_samp))
                lam_ig = np.square(vw['lambda_t'])
                omega_presamp = np.random.wald(mean=nu_ig, scale=lam_ig)
                vw['omega'] = 1/omega_presamp
            elif omega_prior=='ig':
                shape_g = 0.5+0.5
                #rate_g = jnp.square(vw['lambda_t'])/2 + jnp.square(z_samp)/(2*vw['sigma2'])
                rate_g = 0.5*(1/jnp.square(vw['lambda_t']) + jnp.square(z_samp)/vw['sigma2'])
                vw['omega'] = 1/np.random.gamma(shape_g, 1/rate_g)
            else:
                raise Exception("Unknown omega prior!")

        if 'beta' in fixed:
            #z_samp = B @ (vw['beta'].flatten())
            pass
        else:
            quickboi = True
            if comp_prior=='shear':
                xi_1 = np.random.normal(size=P)
                xi_2 = np.random.normal(size=K*P)
                vw['beta'] = get_beta_samp_shear(grams_all, B, vw['omega'], vw['lambda_p'], Xty[:,:,0], xi_1, xi_2, vw['sigma2'])
            elif quickboi:
                beta_mean = woodbury_mean(B, vw['lambda_p'], vw['omega'], grams_aug_inv_all, bha_aug)
                beta_mean = beta_mean.flatten()
                xi_1 = np.random.normal(size=P)
                xi_2 = np.random.normal(size=K*P)
                vw['beta'] = get_beta_samp(grams_aug_invchols, B, vw['omega'], vw['lambda_p'], xi_1, xi_2, beta_mean, vw['sigma2'])
            else:
                assert PROJ_B
                SIGMA = jnp.linalg.inv(block_diag(*grams_all) + vw['lambda_p']*ImPB + B.T @ jnp.diag(1/vw['omega']) @ B)
                mu = SIGMA @ (Xa.T @ ya)
                Lsig = jnp.linalg.cholesky(SIGMA)
                vw['beta'] = np.sqrt(vw['sigma2']) * Lsig @ np.random.normal(size=K*P) + mu
            z_samp = B @ vw['beta']
            vw['beta'] = vw['beta'].reshape([K,P])

        if np.any(~np.isfinite(vw['beta'])):
            raise Exception("nan beta")

        if np.any(~np.isfinite(z_samp)):
            raise Exception("nan z")

        ## eta updates
        if (K>1) and 'eta' not in fixed and i > eta_freeze:

            # Update all eta chains.
            etas_prop = jnp.copy(etas)
            inds_flip = np.random.choice(K-1,L,replace=True)
            for l in range(L):
                etas_prop = etas_prop.at[l,inds_flip[l]].set(1-etas_prop[l,inds_flip[l]])

            etas_send = np.zeros([2*L,K-1])
            etas_send[0::2,:] = etas
            etas_send[1::2,:] = etas_prop

            # NOTE: Note sure if faster to return all Bvals here or to just recompute later.
            nlp, resids = eta_nld(etas_send, grams_all, Xa, ya, vw, Ns, vw['tau2'])
            nlp = nlp.reshape([L,2])

            temps = d2t(deltas)
            nlpt = nlp / temps[:,None]

            # Metropolis step
            lalpha = -nlpt[:,1] + nlpt[:,0]
            lu = np.log(np.random.uniform(size=L))
            samps = (lalpha > lu).astype(int)

            etas_accepts[i,:] = samps
            samp = samps[0]

            nlpa = np.array([nlp[l,samps[l]] for l in range(L)])

            etas = np.array([etas_send.reshape([L,2,K-1])[l,samps[l],:] for l in range(L)])

            ##
            # Try switching one of them.
            if L > 1:
                ll = np.random.choice(L-1,1)[0]
                who[i] = ll

                lpnew = -nlpa[ll+1]/temps[ll] - nlpa[ll]/temps[ll+1]
                lpold = -nlpa[ll]/temps[ll] - nlpa[ll+1]/temps[ll+1]
                lbeta = lpnew - lpold
                lu = np.log(np.random.uniform())
                isswitch = lbeta > lu
                didswitch[i] = isswitch
                if isswitch:
                    temp = np.copy(etas[ll,:])
                    etas[ll,:] = etas[ll+1,:]
                    etas[ll+1,:] = temp

                ## NOTE: Adaptation.
                ## Adjust proposal var.
                #print(f"Old delta: {}")
                beta = np.minimum(np.exp(lbeta),1.)
                gamma_it = np.power(1./(i+1),gamma_exp)
                delta_pre = deltas[ll] + gamma_it * (beta - desired_sr)
                #deltas[ll] = np.minimum(deltas[ll]+2, np.maximum(deltas[ll]-2, delta_pre))
                deltas[ll] = delta_pre
                delta_tracking[i,:] = deltas 
                ## NOTE: Adaptation.

            vw['eta'] = etas[0,:]

            B = get_B(grams_all, vw['eta'][None,:], Ns, vw['tau2'])[0,:,:] # TODO: is recalculating here faster than just returning earlier?
        else:
            _, resids = eta_nld(vw['eta'][None,:], grams_all, Xa, ya, vw, Ns, vw['tau2'])
            samp = 0

        if K > 1:
            beta_cB = resids[samp,:]
            z_samp = B @ vw['beta'].flatten()
        else:
            beta_cB = vw['beta']
            z_samp = np.zeros(P)

        ## lam updates
        if 'lambda_t' in fixed:
            pass
        else:
            A = 1.#  Cauchy scale.
            if omega_prior=='exp':
                phi_samp = 1/jnp.square(vw['lambda_t'])
                a_samp = 1/np.random.gamma(1.,1/(A+1/phi_samp))
                phi_samp = 1/np.random.gamma(P+1/2,1/(1/a_samp + 0.5*jnp.sum(vw['omega'])))
                vw['lambda_t'] = 1/jnp.sqrt(phi_samp)
            elif omega_prior=='ig':
                phi_samp = np.square(vw['lambda_t'])
                a_samp = 1/np.random.gamma(1.,1/(A+1/phi_samp)) #TODO: DRY violation.
                g_rate = 1/a_samp + 0.5*jnp.sum(1/vw['omega'])
                phi_samp = 1/np.random.gamma((P+1)/2,1/g_rate)
                vw['lambda_t'] = jnp.sqrt(phi_samp)
            else:
                raise Exception("Unknown omega_prior.")
        if 'lambda_p' in fixed:
            pass
        else:
            phi_samp = 1/vw['lambda_p']
            A = 1. # Cauchy scale (or inverse-scale?).
            a_samp = 1/np.random.gamma(1.,1/(A+1/phi_samp))
            beta_proj_norm2 = jnp.sum(jnp.square(beta_cB)) / vw['sigma2'] # TODO:Make sure not to copy this last part when moving.
            dof = (K-1)*P
            phi_samp = 1/np.random.gamma(dof/2 + 0.5, 1/(beta_proj_norm2/2 + 1/a_samp))
            vw['lambda_p'] = 1/phi_samp

        if 'sigma2' in fixed:
            pass
        else:
            shape_prec = (Na+P*K) / 2
            sse = jnp.sum(jnp.square(ya - Xa @ vw['beta'].flatten()))
            ssp1 = jnp.sum(jnp.square(z_samp) / vw['omega'])
            beta_proj_norm2 = jnp.sum(jnp.square(beta_cB)) 
            ssp2 = beta_proj_norm2*vw['lambda_p']
            scale_prec = 1/((sse+ssp1+ssp2)/2)
            vw['sigma2'] = 1/np.random.gamma(shape=shape_prec, scale = scale_prec)

        if 'rho' in fixed:
            pass
        else:
            rho_a = 1.
            rho_b = 1.
            rho_alpha = rho_a + np.sum(vw['eta'])
            rho_beta = rho_b + (K-1) - np.sum(vw['eta'])
            vw['rho'] = np.random.beta(rho_alpha,rho_beta)

        if 'tau2' in fixed:
            pass
        else:
            tau2_pd = 'rw_normal'
            symmetric_proposals = ['rw_normal','rw_cauchy']
            if tau2_pd=='rw_cauchy':
                tau2_prop = vw['tau2'] + tau2_prop_scale*np.random.standard_cauchy()
            elif tau2_pd=='rw_normal':
                tau2_prop = vw['tau2'] + tau2_prop_scale*np.random.normal()
            elif tau2_pd=='unif':
                tau2_prop = np.power(10.,np.random.uniform(-2,6))
            else:
                raise Exception("Unknown tau2 prop dist!")

            cur_nll, _ = eta_nld(vw['eta'][None,:], grams_all, Xa, ya, vw, Ns, vw['tau2'])
            if tau2_prop > 0:
                prop_nll, _ = eta_nld(vw['eta'][None,:], grams_all, Xa, ya, vw, Ns, tau2_prop)
            else:
                prop_nll = [np.inf]

            cur_lpd = tfpd.Cauchy(0,1).log_prob(vw['tau2'])
            prop_lpd = tfpd.Cauchy(0,1).log_prob(tau2_prop)

            if tau2_pd in symmetric_proposals:
                back_lprb = 0.
                prop_lprb = 0.
            elif tau2_pd=='unif':
                back_lprb = np.log10(vw['tau2'])
                prop_lprb = np.log10(tau2_prop)
            else:
                raise Exception("Unknown tau2 prop dist!")
 
            met_num = -prop_nll[0] + prop_lpd + back_lprb
            met_den = -cur_nll[0] + cur_lpd + prop_lprb

            if not np.isfinite(met_den):
                lalpha = -np.inf
            else:
                lalpha = (met_num - met_den)

            ## NOTE: MH Step
            #debug_tau2 = True
            debug_tau2 = False
            lu = np.log(np.random.uniform())
            isaccept = lalpha > lu
            if isaccept:
                if debug_tau2:
                    print("tau2 Accept!")
                tau2_accepts += 1
                vw['tau2'] = tau2_prop
            else:
                if debug_tau2:
                    print("tau2 Reject!")
                tau2_rejects += 1
            if debug_tau2:
                print(f"tau2 lalpha: {lalpha}")
            ## NOTE: MH Step

            ## NOTE: Adaptation.
            ## Adjust proposal var.
            alpha = np.minimum(np.exp(lalpha),1.)
            gamma_it = np.power(1./(i+1),gamma_exp)
            tau2_prop_scale = np.exp(np.log(tau2_prop_scale) + gamma_it * (alpha - desired_ar))
            if debug_tau2:
                print(f"tau2 prop scale: {tau2_prop_scale}")
                print(f"tau2 prop was: {tau2_prop}")
                print(f"mean eta val: {np.mean(vw['eta'])}")
            ## NOTE: Adaptation.

        if ('psi' in fixed) or (not docorr):
            pass
        else:
            def lp_psi(psi, vw):
                def get_Lk(psi, k):
                    print("delete this pls.")
                    err_mat = np.linalg.inv(psi*err_Cs[k]+np.eye(Ns[k]))
                    Lk = np.linalg.cholesky(err_mat)
                    return Lk
                qf = 0
                logdet = 0
                for k in range(len(Xs)):
                    rk = ys[k] - Xs[k]@vw['beta'][k,:]
                    Lk = get_Lk(psi, k)
                    qf += np.sum(np.square(Lk@rk))/(2*vw['sigma2'])
                    logdet += np.sum(np.log(np.diag(Lk)))
                ll = -qf + logdet
                lpri = np.log(psi)
                lp = ll + lpri
                return float(lp)

            psi_prop = vw['psi'] + psi_prop_scale*np.random.normal()

            cur_ld = lp_psi(vw['psi'], vw)
            if psi_prop > 0:
                prop_ld = lp_psi(psi_prop, vw)
            else:
                prop_ld = -np.inf

            # From symmetry of proposal.
            back_lprb = 0.
            prop_lprb = 0.
 
            met_num = prop_ld + back_lprb
            met_den = cur_ld + prop_lprb

            if not np.isfinite(met_den):
                lalpha = -np.inf
            else:
                lalpha = (met_num - met_den)

            ## NOTE: MH Step
            debug_psi = True
            #debug_psi = False
            lu = np.log(np.random.uniform())
            isaccept = lalpha > lu
            if isaccept:
                if debug_psi:
                    print("psi Accept!")
                psi_accepts += 1
                vw['psi'] = psi_prop
            else:
                if debug_psi:
                    print("psi Reject!")
                psi_rejects += 1
            if debug_psi:
                print(f"psi lalpha: {lalpha}")
            ## NOTE: MH Step

            ## NOTE: Adaptation.
            ## Adjust proposal var.
            alpha = np.minimum(np.exp(lalpha),1.)
            gamma_it = np.power(1./(i+1),gamma_exp)
            psi_prop_scale = np.exp(np.log(psi_prop_scale) + gamma_it * (alpha - desired_ar))
            if debug_psi:
                print(f"psi prop scale: {psi_prop_scale}")
                print(f"psi prop was: {psi_prop}")
                print(f"mean eta val: {np.mean(vw['eta'])}")
            ## NOTE: Adaptation.

        # Update traces.
        for v in tracking:
            tracking[v][i] = jnp.copy(vw[v])

    # Traceplot
    if plotprefix is not None:
        toplot = 4
        nvars = [int(np.prod(tracking[v].shape[1:])) for v in tracking]
        varseach = [np.minimum(toplot,nvars[vi]) for vi,v in enumerate(tracking)]
        plotseach = [2*varseach[vi] for vi,v in enumerate(tracking)]
        nplots = np.sum(plotseach)
        ncols = 4
        nrows = int(np.ceil(nplots/ncols))
        plt.figure(figsize=[1.5*ncols,1.5*nrows])
        plottrans = {'sigma2' : np.log10, 'lambda_t' : np.log10, 'lambda_p' : np.log10, 'omega' : np.log10, 'tau2' : np.log10, 'psi' : np.log10}
        for vi,v in enumerate(tracking):
            samps = tracking[v].reshape([iters,nvars[vi]])
            trans = plottrans[v] if v in plottrans else lambda x: x
            for p in range(varseach[vi]):
                title = fr"$\{v}^{p}$"
                plt.subplot(nrows,ncols,1+int(np.sum(plotseach[:vi]))+2*p)
                plt.hist(trans(samps[burnin:,p]))
                if true_vals is not None:
                    ll,ul = plt.gca().get_ylim()
                    try:
                        tv = true_vals[v][p]
                    except Exception:
                        tv=true_vals[v]
                    tv = trans(tv)
                    plt.vlines(tv,ll,ul,color='orange',linestyle='--')
                plt.title(title)
                plt.subplot(nrows,ncols,1+int(np.sum(plotseach[:vi]))+2*p+1)
                plt.plot(trans(samps[:,p]))
                ll,ul = plt.gca().get_ylim()
                plt.vlines(burnin,ll,ul,color='gray',linestyle='--')
                if true_vals is not None:
                    ll,ul = plt.gca().get_xlim()
                    plt.hlines(tv,ll,ul,color='orange',linestyle='--')
                plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{plotprefix}_plot.pdf")
        plt.close()

        # Visualize dataset indicator trajectory.
        # Not exactly MCA because that seems to give weird results here.
        #Z = tracking['eta'] / (np.sum(tracking['eta'])+1e-3)
        #r = np.mean(Z,axis=1) + 1/Z.shape[0]
        #c = np.mean(Z,axis=0) + 1/Z.shape[1]
        #tosvd = np.diag(1/np.sqrt(r)) @ (Z - r[:,None,]@c[None,:]) @ np.diag(1/np.sqrt(c))
        mu = np.mean(tracking['eta'],axis=0)
        sig = np.std(tracking['eta'], axis=0)+1e-6
        tosvd = (tracking['eta'] - mu[None,:]) / sig[None,:]
        Zsvd = np.linalg.svd(tosvd,full_matrices=False)
        projr = Zsvd[2].T[:,:2]
        z = tosvd @ projr

        nrand = 20
        rand = np.random.binomial(1,0.5,size=[nrand,K-1])
        rand = (rand - mu[None,:]) / sig[None,:]
        zrand = rand @ projr

        bound = np.stack([np.zeros(K-1),np.ones(K-1)])
        bound = (bound - mu[None,:]) / sig[None,:]
        zbound = bound @ projr

        # Yellow is Later (yater)
        if K > 2:
            fig = plt.figure()
            zj = z #+ np.std(z,axis=0)[None,:]*1e-1*np.random.normal(size=z.shape)
            plt.scatter(zj[:,0],zj[:,1], color = plt.colormaps['autumn'](np.arange(iters)/iters))
            plt.scatter(zrand[:,0],zrand[:,1], color = 'blue', label = 'rand')
            plt.scatter(zbound[0,0],zbound[0,1], color = 'cyan', label = 'zeros')
            plt.scatter(zbound[1,0],zbound[1,1], color = 'purple', label = 'ones')
            plt.legend()
            plt.savefig(f"{plotprefix}_inds.pdf")
            plt.close()

        if L > 1:
            plt.figure()
            temps_tracking = np.apply_along_axis(d2t, 1, delta_tracking)
            for l in range(L):
                p = plt.plot(temps_tracking[:,l])
                plt.hlines(temps_tracking[0,l], 0, iters, color = p[0].get_color(), linestyle = '--')
            plt.yscale('log')
            plt.savefig(f"{plotprefix}_switch.pdf")
            plt.close()

    for v in tracking:
        tracking[v] = np.take(tracking[v], np.arange(burnin,iters), 0)

    if not 'tau2' in fixed:
        print(f"tau2 rate: {tau2_accepts / (tau2_accepts+tau2_rejects)}")

    switch_rates = []
    for l in range(L-1):
        switch_rates.append(np.mean(didswitch[who==l]))

    if verbose:
        print("Accept rates:")
        print(np.mean(etas_accepts,axis=0))
        print("Switching rates:")
        print(switch_rates)

    beta0_hat = np.mean(tracking['beta'], axis = 0)[0,:]

    return beta0_hat, tracking, switch_rates

###### Carlin-Polson-style sampler.
@jax.jit
def get_beta_samp(grams_invchols, B, omega_samp, lambda_samp, xi_1, xi_2, beta_mean, sigma2_samp):
    K = grams_invchols.shape[0]
    P = grams_invchols.shape[1]
    LiBT = (grams_invchols @ B.transpose([1,0]).reshape([K,P,P])).reshape([K*P,P])

    svLiB = jnp.linalg.svd(LiBT, full_matrices=False)
    U1_pre = svLiB[0]
    Sigma = jnp.diag(svLiB[1])
    V = svLiB[2].T
    BBT = B@B.T
    BBTpi = jnp.linalg.inv(BBT)
    DELTA = Sigma @ (V.T @ (jnp.diag(1/omega_samp)- lambda_samp*BBTpi)@V)@Sigma
    lamd, W = jnp.linalg.eigh(DELTA)
    U1 = U1_pre @ W

    D = jnp.diag(1/jnp.sqrt(1+lamd))
    a1 = U1 @ (D @ xi_1)
    a2 = xi_2 - U1 @ (U1.T @ xi_2)
    a = a1 + a2
    LiTa = (jnp.matrix_transpose(grams_invchols) @ a.reshape([K,P,1])).flatten()
    beta_working = jnp.sqrt(sigma2_samp) * LiTa + beta_mean

    return beta_working

@jax.jit
def get_beta_samp_shear(grams_all, B, omega_samp, lambda_samp, Xty, xi_1, xi_2, sigma2_samp):
    """
    Draw beta | rest under the shear prior. Sheared coordinates gamma = (z, beta_S), z = beta_0 - T beta_S:
        gamma ~ N(Q^{-1} b, sigma2 Q^{-1}),  Q = A + C^T G_0 C,  A = blockdiag(Omega^{-1}, G_k + lambda_p I),
        C = [I_P, T],  b = C^T X_0^T y_0 + [0; X_k^T y_k].
    With A = L L^T and M = L^{-1} C^T = U S V^T, DELTA = S V^T G_0 V S = W diag(l) W^T, U1 = U W:
        Q^{-1} = L^{-T} (I - U1 diag(l/(1+l)) U1^T) L^{-1},
    so mean and draw share one KPxP SVD and one PxP eigendecomposition (cf. Bhattacharya et al. 2016;
    same structure as get_beta_samp but with W = G_0, and no pinv(BB^T)). Returns beta flattened (KP,).
    Passing grams_all = 0 and Xty = 0 draws from the prior.
    """
    K = grams_all.shape[0]
    P = grams_all.shape[1]
    G0 = grams_all[0]
    T = -B[:,P:]                                                     # P x (K-1)P
    Tk = T.reshape([P,K-1,P]).transpose([1,0,2])                     # [K-1,P,P]
    A_src = grams_all[1:] + lambda_samp*jnp.eye(P)[None,:,:]
    L_src = jnp.linalg.cholesky(A_src)
    Linv_src = jnp.linalg.inv(L_src)                                 # [K-1,P,P]
    sqo = jnp.sqrt(omega_samp)                                       # L_0^{-1} = diag(sqrt(omega))

    # M = L^{-1} C^T, stacked [KP, P]
    M = jnp.concatenate([jnp.diag(sqo)[None,:,:], Linv_src @ jnp.matrix_transpose(Tk)], axis=0).reshape([K*P,P])
    svM = jnp.linalg.svd(M, full_matrices=False)
    U, S, Vt = svM[0], svM[1], svM[2]
    DELTA = (S[:,None]*Vt) @ G0 @ (Vt.T*S[None,:])
    lamd, Wg = jnp.linalg.eigh(DELTA)
    U1 = U @ Wg

    # b and L^{-1} b
    x0 = Xty[0]
    bk = jnp.matrix_transpose(Tk) @ x0[None,:,None] + Xty[1:][:,:,None]  # [K-1,P,1]
    Lib = jnp.concatenate([(sqo*x0)[None,:,None], Linv_src @ bk], axis=0).flatten()

    # w = (I - U1 diag(l/(1+l)) U1^T) L^{-1} b  +  sqrt(sigma2) * [ U1 (1+l)^{-1/2} xi_1 + (I - U1 U1^T) xi_2 ]
    w_mean = Lib - U1 @ ((lamd/(1+lamd)) * (U1.T @ Lib))
    a = U1 @ (xi_1/jnp.sqrt(1+lamd)) + xi_2 - U1 @ (U1.T @ xi_2)
    w = w_mean + jnp.sqrt(sigma2_samp)*a

    # gamma = L^{-T} w, then un-shear.
    z = sqo*w[:P]
    beta_S = (jnp.matrix_transpose(Linv_src) @ w[P:].reshape([K-1,P,1])).flatten()
    beta_0 = z + T @ beta_S
    return jnp.concatenate([beta_0, beta_S])

@jax.jit
def woodbury_mean(B, lambda_p, omega, grams_aug_inv_all, bha_aug):
    K = grams_aug_inv_all.shape[0]
    P = grams_aug_inv_all.shape[1]
    b1 = B @ bha_aug.flatten()
    Ba = B.reshape([P,K,P]).transpose([1,0,2])
    # TODO: is cholesky faster?
    BXtXiBt = jnp.sum(Ba @ grams_aug_inv_all @ jnp.matrix_transpose(Ba),axis=0)
    S = jnp.linalg.inv(jnp.diag(1/omega)-lambda_p*jnp.linalg.pinv(B@B.T)) + BXtXiBt 
    b2 = jnp.linalg.solve(S, b1)
    b3 = B.T @ b2
    b4 = grams_aug_inv_all @ b3.reshape([K,P,1])
    b5 = bha_aug - b4

    return b5

@jax.jit
def get_B(grams_all, etas, Ns, tau2):
    K = grams_all.shape[0]
    P = grams_all.shape[1]
    E0 = jnp.concatenate([jnp.eye(P)]+[jnp.zeros([P,P]) for _ in range(K-1)],axis=1)

    grams_rel = grams_all[None,1:,:,:]*etas[:,:,None,None]
    cat_grams = jnp.matrix_transpose(grams_rel).reshape([etas.shape[0],(K-1)*P,P]).transpose([0,2,1])
    sum_grams = jnp.sum(grams_rel,axis=1)
    rel_Ns = jnp.sum(Ns[None,1:]*etas, axis = 1)
    # TODO: Question: is this better with Ns or without?
    sum_grams += tau2*jnp.eye(grams_all.shape[1])[None,:,:]  * rel_Ns[:,None,None]
    #sum_grams = sum_grams.at[jnp.where(jnp.sum(etas,axis=1)==0)[0],:,:].set(jnp.eye(P,P))# a hack to make T = 0 if eta = 0.
    sum_grams = sum_grams + (jnp.sum(etas,axis=1)==0)[:,None,None]*jnp.eye(P)[None,:,:]
    T = jnp.linalg.solve(sum_grams,cat_grams)
    T = jnp.concatenate([jnp.zeros([etas.shape[0],P,P]),T],axis=2)
    B = E0[None,:,:] - T
    return B


