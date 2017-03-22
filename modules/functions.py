"""functions.py: functions used by itebd.py

"""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh,LinearOperator,eigs
import warnings
import sys
import pickle
from modules import SU2kMPS_file, SU2k_data

########
# init #
########

def init_itebd(pars):
    """init itebd either from file or by computing 2 anyons problem

    """
    # Open file previous trunc_D and previous eps and dt
    min_D = 4
    my_pars = pars.copy()
    trunc_D = pars['chi']
    for cur_D in range(trunc_D, min_D - 1, -1):
        eps = pars['eps']
        while eps < 1e-04:
            dt = pars['dt']
            while dt < 0.1:
                file_name = 'init/SU2k_itebd_'
                file_name += 'hmax_'+str(pars['hmax'])+'_'
                file_name += 'chi_'+str(cur_D)+'_'
                file_name += 'dt_'+str(dt)+'_'
                file_name += 'eps_'+'{:.0e}'.format(eps)
                try:
                    # Use numpy routines
                    with open(file_name, 'rb') as f:                    
                        A = pickle.load(f)
                        Lambda = pickle.load(f)
                        B = pickle.load(f)
                        Lambda_old = pickle.load(f)
                        print('init_itebd: from file', file_name)
                        return A, Lambda, B, Lambda_old
                except IOError:
                    pass
                dt *= 10
            eps *= 10

    # If here it means that no file has been found, and curD = min_D
    # so init_itebd_4 is called
    print('init_itebd: from 2 anyons')
    A, Lambda, B, Lambda_old = init_itebd_4(pars['hmax'])

    return A, Lambda, B, Lambda_old

def init_itebd_mod(pars):
    """init itebd either from file or by computing 2 anyons problem

    """
    # Open file previous trunc_D and previous eps and dt

    min_D = 4
    my_pars = pars.copy()
    trunc_D = pars['chi']
    for cur_D in range(trunc_D, min_D - 1, -1):
        eps = pars['eps']
        while eps < 1e-04:
            dt = pars['dt']
            # TODO: Use the function read_from_file_mod
            while dt < 0.1:
                file_name = 'init/SU2k_itebd_'
                file_name += 'hmax_'+str(pars['hmax'])+'_'
                file_name += 'a_'+str(pars['height0'])+'_'
                file_name += 'b_'+str(pars['heightL'])+'_'
                file_name += 'chi_'+str(cur_D)+'_'
                file_name += 'dt_'+str(dt)+'_'
                file_name += 'eps_'+'{:.0e}'.format(eps)
                try:
                    with open(file_name, 'rb') as f:                    
                        AA = pickle.load(f)
                        lA = pickle.load(f)
                        AB = pickle.load(f)
                        lB = pickle.load(f)
                        A1 = pickle.load(f)
                        A2 = pickle.load(f)
                        Lt = pickle.load(f)
                        B1 = pickle.load(f)
                        B2 = pickle.load(f)
                        print('init_itebd: from file', file_name)
                        return lA,lB,AA,AB
                except IOError:
                    pass
                dt *= 10
            eps *= 10

    # If here it means that no file has been found, and curD = min_D
    # so init_itebd_4 is called
    print('init_itebd: from 2 anyons')
    A, Lambda, B, Lambda_old = init_itebd_4(pars['hmax'],\
                                            pars['height0'],pars['heightL'])
    lA,lB,AA,AB = orig_to_mod(A,Lambda,B,Lambda_old)
    return lA,lB,AA,AB

def mod_to_orig(lA,lB,AA,AB):
    A=lB.diag().contract(AA).contract(lA.diag().matrix_inv())
    L=lA
    B=AB
    Lold=lB
    return A,L,B,Lold

def orig_to_mod(A,Lambda,B,Lambda_old):
    lA=Lambda
    lB=Lambda_old
    AA=lB.diag().matrix_inv()
    AA=AA.contract(A)
    AA=AA.contract(lA.diag())
    AB=B
    return lA,lB,AA,AB

def init_itebd_4(hmax,a,b):
    """Returns A,L,B,L_old from an alternating path or better, the exact
    ground state of 4 SU2k anyons of charge 2. Hamiltonian supposed to
    be -e1-e2-e3; e_i is TL generator.

    a,b are the left/right boundary heights.

    We consider only even number of anyons -> parity a=parity b.

    """
    if a < 1 or a > hmax or b < 1 or b > hmax:
        print("In init_itebd_4: not valied bc",a,b)
        sys.exit(1)
    if a!=b:
        print("this is an excited state...")
        sys.exist(1)
    
    # init to the ground state of 2 anyons system:
    if a == 1:
        init_shape = [[1],[1],[1],[1],[1]]
        init_heights = [[a],[2],[a+1],[2],[a]]
        init_sects = {}
        init_sects[a,2,a+1,2,b] = np.array([1],dtype=float).reshape(1,1,1,1,1)
    elif a == hmax:
        init_shape = [[1],[1],[1],[1],[1]]
        init_heights = [[a],[2],[a-1],[2],[a]]
        init_sects = {}
        init_sects[a,2,a-1,2,a] = np.array([1],dtype=float).reshape(1,1,1,1,1)
    else: 
        # eigenvector of matrix [[x,sqrt(x y)],[sqrt(x y),y]] with
        # x=-SU2k_data.qpsi(hmax,a-1)/SU2k_data.qpsi(hmax,a)
        # y=-SU2k_data.qpsi(hmax,a+1)/SU2k_data.qpsi(hmax,a)
        # so its coefficients are:
        Cam1 = -np.sqrt(SU2k_data.qpsi(hmax,a-1))
        Cap1 = np.sqrt(SU2k_data.qpsi(hmax,a+1)) 
        norm = np.sqrt(Cam1**2 + Cap1**2)
        Cam1 = Cam1/norm
        Cap1 = Cap1/norm
        init_shape = [[1],[1],[1,1],[1],[1]]
        init_heights = [[a],[2],[a-1,a+1],[2],[a]]
        init_sects = {}
        init_sects[a,2,a-1,2,a] = np.array([Cam1],dtype=float).reshape(1,1,1,1,1)
        init_sects[a,2,a+1,2,a] = np.array([Cap1],dtype=float).reshape(1,1,1,1,1)

    init = SU2kMPS_file.SU2kMPS(shape=init_shape, heights=init_heights,
                                sects=init_sects, hmax=hmax, dtype=float)
    A,L,B,rel_err = init.svd_2_sites()
    # Lambda_old
    L_old = SU2kMPS_file.SU2kMPS([[1]], heights=[[a]], hmax=hmax,
                                      invar=False, dtype=float)
    L_old[(a,)] = np.array([1])
    
    return A,L,B,L_old

    
#############
# Operators #
#############

def act_id(psi):
    """identity: Just return psi

    """
    return psi

def act_order_par(psi, m, i=1):
    """Act with the order parameter of pasquier at position i:
    phi_m(i) = sum(qV_m^a/qpsi(a) proj(h_i=a), a). Index as

        __|__|__|__|__|__

    i =  0| 1| 2| 3| 4| 5

    """
    # checks:
    assert(psi.is_wf())
    first_ind = 0
    last_ind = len(psi.shape)-1
    ind_i = 2*i # index corresponding to i in the SU2kMPS
    if ind_i not in range(first_ind, last_ind + 1, 2):
        print("In act_order_par: wrong index ind_i", ind_i)
        print("not in range", range(first_ind, last_ind + 1, 2))
        sys.exit(1)
    hmax = psi.hmax
    if m not in range(1,hmax+1):
        print("In act_order_par: value of m",m," not in 1,...,",hmax)
        sys.exit(1)
    # compute the new sectors: just multiply old ones by weight
    res = psi.empty_like()
    for k,v in psi.sects.items():
        hi = k[ind_i]
#        print('hi',hi,'weight',SU2k_data.order_par_weight(hmax, m, hi))
        res.sects[k] = SU2k_data.order_par_weight(hmax, m, hi)*v
    # Set the new SU2kMPS and return:
    return res

# Due to the way we defined transfer operator, we cannot pass m as an
# arguement.  So just define several functions for each m.
def act_order_par_m1(psi, i=1): 
    return act_order_par(psi, 1, i)
def act_order_par_m2(psi, i=1):
    return act_order_par(psi, 2, i)
def act_order_par_m3(psi, i=1):
    return act_order_par(psi, 3, i)
def act_order_par_m4(psi, i=1):
    return act_order_par(psi, 4, i)
def act_order_par_m5(psi, i=1):
    return act_order_par(psi, 5, i)
def act_order_par_m6(psi, i=1):
    return act_order_par(psi, 6, i)

def act_S1(psi,i=1):
    """Act with e_i + e_{i+1} on wf

    """
    assert(psi.is_wf())
    psi1 = act_TL_gen(psi, i=i)
    psi2 = act_TL_gen(psi, i=i+1)
    return psi1+psi2

def act_S2(psi,i=1):
    """Act with 1/2 S1(i)+1/2 S1(i+1)
    1/2(e_i + e_{i+1}) + 1/2(e_{i+1} + e_{i+2}) on wf

    """
    assert(psi.is_wf())
    psi1 = act_S1(psi, i=i)
    psi2 = act_S1(psi, i=i+1)
    return 1/2.*(psi1+psi2)

def act_D1(psi,i=1):
    """Act with e_i - e_{i+1} on wf

    """
    assert(psi.is_wf())
    psi1 = act_TL_gen(psi, i=i)
    psi2 = act_TL_gen(psi, i=i+1)
    return psi1-psi2

def act_R(psi,i=1):
    """Act with [e_i , e_{i+1}] on wf

    """
    assert(psi.is_wf())
    psi1 = act_TL_gen(psi, i=i+1)
    psi1 = act_TL_gen(psi1, i=i)
    psi2 = act_TL_gen(psi, i=i)
    psi2 = act_TL_gen(psi2, i=i+1)
    return psi1-psi2

def act_R1(psi,i=1):
    """Act with 1/2([e_i,e_{i+1}] + [e_{i+1},e_{i+2}]) on wf

    """
    assert(psi.is_wf())
    psi1 = act_R(psi, i=i)
    psi2 = act_R(psi, i=i+1)
    return 1/2.*(psi1+psi2)

def act_F1(psi,i=1):
    """Act with [e_i,e_{i+1}] - [e_{i+1},e_{i+2}] on wf

    """
    assert(psi.is_wf())
    psi1 = act_R(psi, i=i)
    psi2 = act_R(psi, i=i+1)
    return psi1-psi2

def act_TL_gen(psi, i=1):
    """Act with the Temperley Lieb generator on a SU2kMPS with shape =
    [[n1,...],[1],[1,...],[1],[n3,...]]  heights =
    [[h1,...],[2],[h2,...],[2],[h3,...]]  to produce another
    SU2kMPS with of the same type with sects containing the result
    of the action of e_2

    i = index of TL, default i=1:

        __|__|__|__|__|__

    i =  0| 1| 2| 3| 4| 5

    """
    # checks:
    assert(psi.is_wf())
    first_ind = 0
    last_ind = len(psi.shape)-1
    ind_i = 2*i # index corresponding to i in the SU2kMPS
    ind_im1 = 2*(i-1)
    ind_ip1 = 2*(i+1)
    if ind_i not in range(first_ind + 2, last_ind - 1):
        print("In act_TL_gen: wrong index ind_i", ind_i)
        print("not in range",range(first_ind + 2, last_ind - 1))
        sys.exit(1)       
    # compute the new sectors
    res_sects = {}
    set_hi_new = set() # set so that only unique values are retained
    for k,v in psi.sects.items():
        him1 = k[ind_im1]
        hi = k[ind_i]
        hip1 = k[ind_ip1]
        if him1 == hip1:
            for hi_new in SU2k_data.fusion_range(psi.hmax, him1, 2):
                set_hi_new.add(hi_new)
                new_k = list(k)
                new_k[ind_i] = hi_new
                new_k = tuple(new_k)
                if new_k in res_sects:
                    # add, implementing the sum over hi which contributes
                    # to the new wavefunction of height hi_new
                    res_sects[new_k] += v*SU2k_data.TL_weight(psi.hmax, him1,
                                                              hi, hi_new)
                else: #init
                    res_sects[new_k] = v*SU2k_data.TL_weight(psi.hmax, him1,
                                                             hi, hi_new)
        # else (him1 != hip1), continue without setting res_sects,
        # which is then zero.
                    
    list_hi_new = sorted(list(set_hi_new))
    new_dim_ind_i = [1] * len(list_hi_new)
    res_shape = psi.shape[:ind_i]+[new_dim_ind_i]+psi.shape[ind_i+1:]
    res_heights = psi.heights[:ind_i]+[list_hi_new]+psi.heights[ind_i+1:]
    # Set the new SU2kMPS and return:
    res_shape, res_heights = psi.sorted_shape_heights(shape=res_shape, 
                                                       heights=res_heights)
    res = type(psi)(res_shape, heights=res_heights,
                    hmax=psi.hmax, sects=res_sects,
                    dtype=psi.dtype, charge=psi.charge,
                    defval=psi.defval, invar=psi.invar)
    return res

def act_U_bond(beta, dt, psi):
    """Act on psi, an SU2kMPS of the form:
    shape =[[n1,...],[1],[1,...],[1],[n3,...]]  
    heights = [[h1,...],[2],[h2,...],[2],[h3,...]]  
    to produce another SU2kMPS with of the same type 
    with sects containing the result of the action of 
    exp(+dt e_2) (h_i = -e_i),
    where e_2 is TL gen and dt is an argument.
    beta = qdim(hmax,2) is weight of loops
    
    """
    a = 1
    b = 1/beta * (np.exp(dt*beta)-1)
    # first act with TL
    res = b*act_TL_gen(psi)
    # Note: TL can create new heights for middle entry of psi.
    # To take this into acccount we define new_psi which has 
    # same sects as psi but same shape and height as res:
    new_psi = res.empty_like()
    new_psi.sects = psi.sects
    # finally add a*new_psi to res and return    
    return res + a*new_psi

#############
# recursion #
#############

def recursion(A,Lambda,B,Lambda_old,max_it,pars,save):
    """Run the recursion. If the naive entropy has converged, run until
    the true entropy has converged.  This avoids computing the mixed
    representation which is costly in the first few iterations.

    max_it is maximum number of iterations. Return the converged
    A,Lambda,B,Lambda_old

    """
    print('Recursion starts...')


    ##DEBUG
    lA,lB,AA,AB = orig_to_mod(A,Lambda,B,Lambda_old)
    At,Lt,Bt,Loldt=mod_to_orig(lA,lB,AA,AB)
    print("A-At", A-At)
    print("B-Bt", B-Bt)
    print("L-Lt", Lambda-Lt)
    print("Lold-Loldt", Lambda_old-Loldt)

    
    beta = SU2k_data.qdim(pars['hmax'],2)
    eps = pars['eps']
    delta_t = pars['dt']
    trunc_D = pars['chi']
    EEnt_naive_prev = 0
    EEnt_naive_conv = False
    for n in range(max_it):
        # print("***************************")
        print("Iteration num ", n)
        A,Lambda,B,Lambda_old=update_matrices(A, Lambda, B, Lambda_old,
                                              beta, delta_t, trunc_D)
                
    return A,Lambda,B,Lambda_old

def recursion_mod(lA,lB,AA,AB,max_it,pars,save):
    """Run the recursion, modified algorithm (Hasting j math phys
    2009). If the naive entropy has converged, run until the true
    entropy has converged.  This avoids computing the mixed
    representation which is costly in the first few iterations.

    max_it is maximum number of iterations. Return the converged
    lA,lB,AA,AB

    """
    print('Recursion modified starts...')
    beta = SU2k_data.qdim(pars['hmax'],2)
    eps = pars['eps']
    delta_t = pars['dt']
    trunc_D = pars['chi']
    conv = False
    A1=0;A2=0;Lt=0;B1=0;B2=0
    for n in range(max_it):
        lA,lB,AA,AB=update_matrices_mod(lA,lB,AA,AB,
                                        beta, delta_t, trunc_D)

    if max_it == 0 or max_it == 1 or max_it == 2:
        return lA,lB,AA,AB,A1,A2,Lt,B1,B2
    else:
        print("Recusion ended.")
        one_minus_F = 1-fidelity_mod(lB,AA)
        print("1-F=",one_minus_F)
        A1,A2,Lt,B1,B2=mixed_canonical_form_mod(lA,lB,AA,AB,check=True)
        EEnt = get_EEent_mixed_repr(Lt)
        print("EEnt =", EEnt)        
        En_0 = one_point_function(A1,A2,Lt,n_heights=3,parity=0,
                                  act_Op=act_TL_gen)
        En_1 = one_point_function(A1,A2,Lt,n_heights=3,parity=1,
                                  act_Op=act_TL_gen)
        print("Bond energy =", np.mean([En_0,En_1]))        
        
        return lA,lB,AA,AB,A1,A2,Lt,B1,B2

def update_matrices(A, Lambda, B, Lambda_old, beta, delta_t, trunc_D):
    """The main part of recursion

    """
    psi_guess = compute_psi(A, Lambda, B, Lambda_old)
    psi = act_U_bond(beta, delta_t, psi_guess)
    Lambda_old = Lambda
    A, Lambda, B, rel_err = psi.svd_2_sites(chis=trunc_D,
                                            print_errors=0,
                                            break_degenerate=False)

    return A,Lambda,B,Lambda_old

def update_matrices_mod(lA,lB,AA,AB, beta, delta_t, trunc_D):
    """The main part of recursion

    """
    psi = compute_psi_mod(AA,AB)
    C = act_U_bond(beta, delta_t, psi)
    C = C.fuse(inds=(4,3),erase_inds=True)
    C = C.fuse(inds=(0,1))
    #
    tmp=lA.diag().contract(AB)
    psi = tmp.contract(AA)
    
    theta = act_U_bond(beta, delta_t, psi)
    fused_l = theta.fuse(inds=(0,1),erase_inds=True)
    theta = fused_l.fuse(inds=(2,1))

    U, S, V, rel_err, sqrt_sum_S = theta.matrix_svd(chis=trunc_D,
                                        print_errors=0,
                                        break_degenerate=False,
                                        norm_S=False)
    sh = psi.shape
    hs = psi.heights
    AB = V.split(inds=(2,1), new_dim_i = sh[4], new_dim_j = sh[3],
                new_h_i = hs[4], new_h_j = hs[3])
    AA_tmp = C.matrix_dot(V.conj(),transpose_other=True)
    AA = AA_tmp.split(inds=(0,1), new_dim_i = sh[0], new_dim_j = sh[1],
                      new_h_i = hs[0], new_h_j = hs[1])
    lB = lA
    lA = S
    # normalization (Note: in case of idmrg clearly insures that the wave
    # function is normalized)
    lA /= sqrt_sum_S
    AA /= sqrt_sum_S
    
    return lA,lB,AA,AB

def compute_psi_mod(A,B):
    """Returns the modified wave function for 2 sites given
    by B.A
    
    """
    assert(len(A.shape) == 3 and len(B.shape) == 3)
    assert(A.shape[1] == [1] and B.shape[1] == [1])
    return B.contract(A)

def compute_psi(A, L, B, Lold):
    """Returns the wave function for 2 sites given
    the SU2kMPs A,B,L,L_old by contracting:
    psi = L . B . Lold^{-1} . A . L
    
    """
    assert(len(A.shape) == 3 and len(B.shape) == 3)
    assert(A.shape[1] == [1] and B.shape[1] == [1])
    L_mat = L.diag()
    Lold_mat_inv = Lold.diag().matrix_inv()
    # erase_id = True by default, ret is same shape and heights as B in first 2 steps
    tmp_l = L_mat.contract(B) 
    tmp_l = tmp_l.contract(Lold_mat_inv) 
    tmp_r = A.contract(L_mat)
    return tmp_l.contract(tmp_r)

def save_to_file(A, Lambda, B, Lambda_old, pars):
    """Dump to file the matrices using pickle

    """
    # Save data A, Lambda, B, Lambda_old to start next D recurrence
    # from it or for computing entropy, correlators etc
    file_name = 'init/SU2k_itebd_'
    file_name += 'hmax_'+str(pars['hmax'])+'_'
    file_name += 'a_'+str(pars['height0'])+'_'
    file_name += 'b_'+str(pars['heightL'])+'_'
    file_name += 'chi_'+str(pars['chi'])+'_'
    file_name += 'dt_'+str(pars['dt'])+'_'
    file_name += 'eps_'+str(pars['eps'])
    # Use numpy routines
    with open(file_name, 'wb') as f:
        pickle.dump(A, f)
        pickle.dump(Lambda, f)
        pickle.dump(B, f)
        pickle.dump(Lambda_old, f)
    print('save_to_file: saved to ', file_name)

def save_to_file_mod(lA,lB,AA,AB,A1,A2,Lt,B1,B2,pars):
    """Dump to file the matrices using pickle

    """
    # Save data to start next D recurrence from it or for computing
    # entropy, correlators etc
    file_name = 'init/SU2k_itebd_'
    file_name += 'hmax_'+str(pars['hmax'])+'_'
    file_name += 'a_'+str(pars['height0'])+'_'
    file_name += 'b_'+str(pars['heightL'])+'_'
    file_name += 'chi_'+str(pars['chi'])+'_'
    file_name += 'dt_'+str(pars['dt'])+'_'
    file_name += 'eps_'+str(pars['eps'])
    # Use numpy routines
    with open(file_name, 'wb') as f:
        pickle.dump(AA, f)
        pickle.dump(lA, f)
        pickle.dump(AB, f)
        pickle.dump(lB, f)
        pickle.dump(A1, f)
        pickle.dump(A2, f)
        pickle.dump(Lt, f)
        pickle.dump(B1, f)
        pickle.dump(B2, f)

    print('save_to_file_mod: saved to ', file_name)

def read_from_file_mod(pars):
    """Read from file the matrices using pickle

    """
    file_name = 'init/SU2k_itebd_'
    file_name += 'hmax_'+str(pars['hmax'])+'_'
    file_name += 'a_'+str(pars['height0'])+'_'
    file_name += 'b_'+str(pars['heightL'])+'_'
    file_name += 'chi_'+str(pars['chi'])+'_'
    file_name += 'dt_'+str(pars['dt'])+'_'
    file_name += 'eps_'+'{:.0e}'.format(pars['eps'])
    try:
        # Uses the modified itebd matrices
        with open(file_name, 'rb') as f:
            AA = pickle.load(f)
            lA = pickle.load(f)
            AB = pickle.load(f)
            lB = pickle.load(f)
            A1 = pickle.load(f)
            A2 = pickle.load(f)
            Lt = pickle.load(f)
            B1 = pickle.load(f)
            B2 = pickle.load(f)
        print('init: from file', file_name)
    except IOError:
        print('IOError: file ', file_name, ' not found. Exiting.') 
        sys.exit(1)   

    return lA,lB,AA,AB,A1,A2,Lt,B1,B2
    
###################
# Naive functions #
###################

def my_energy(psi):
    assert(psi.is_2_sites)
    psip = act_TL_gen(psi)
    # compute overlap: run over the keys of psi since heights of psi are a subset of the heights of psip
    ret = 0
    for k,v in psi.sects.items():
        vp = psip[k]
        ret += np.tensordot(np.conj(v),vp,axes=([0,4],[0,4])).reshape(1)[0]
    return ret

def my_ord_par(psi,m):
    assert(psi.is_2_sites)
    psip = act_order_par(psi, m, i=1)
    # compute overlap: run over the keys of psi since heights of psi are a subset of the heights of psip
    ret = 0
    for k,v in psi.sects.items():
        vp = psip[k]
        ret += np.tensordot(np.conj(v),vp,axes=([0,4],[0,4])).reshape(1)[0]
    return ret

def my_norm(psi):
    assert(psi.is_2_sites)
    # compute overlap: run over the keys of psi since heights of psi are a subset of the heights of psip
    ret = 0
    for k,v in psi.sects.items():
        ret += np.tensordot(np.conj(v),v,axes=([0,4],[0,4])).reshape(1)[0]
    return ret

##################################################
# Transfer operators and their orthogonalization #
##################################################

def get_E_sects(psi, psip):
    """Fill sectors of transfer operator E from the cell wavefunction 
    E <- psi^* . psip
    
    Used by transfer_operator. Understood that psi.is_wf==psip.is_wf==True

    """
    L = len(psi.shape)
    Lm1 = len(psi.shape)-1
    assert(Lm1 == len(psip.shape)-1)
    assert(psi.compatible_indices(psi, 0, Lm1))
    assert(psi.compatible_indices(psip, 0, 0))
    assert(psi.compatible_indices(psip, Lm1, Lm1))
    E_sects = {}
    odd_inds = list(range(L)[1::2])
    axes=(odd_inds,odd_inds)
    for k,v in psi.sects.items():
        if k in psip.sects:
            vp = psip[k]
            # left and right heights
            hl = k[0]
            hr = k[Lm1]
            new_k = (hl,hr,hl,hr)
            sh = vp.shape
            # these are the only non-trivial shapes by assumption
            new_sh = (sh[0],sh[Lm1],sh[0],sh[Lm1])
            # the tensordot is of 1d indices
            if new_k in E_sects: #implement the sum over intermidiate heights
                E_sects[new_k] += np.tensordot(np.conj(v),vp,
                                               axes=axes).reshape(new_sh)
            else: # init
                E_sects[new_k] = np.tensordot(np.conj(v),vp,
                                              axes=axes).reshape(new_sh)
        # else, just pass, the overlap is zero.
    return E_sects

def transfer_operator(psi_L, psi_R, n_heights=0, parity=0, act_Op=act_id):
    """Returns the transfer operator associated to the unit cell psi_L,
    psi_R with inserted operators whose action is given by *act_Op =
    variable number of arguments.
    
    Ops is a function which implements the action of the Operator on
    psi or compositions thereof.  n_heights is the number of heights
    the operator acts on. n_heights=0 means no operator. parity is if
    it acts on even (0) or odd (1) sites.  Suppose that act_Op has an
    argument i, which is the site on which Op acts, specified by
    parity.

    Note, this is defined as the repeating structure involved
    in computing expectation values in the ground state specified
    by the matrices A,B,L,L_old, and so the heights are as follows:
    (numbers are the order of labeling of indices)
    
    0  h---------h'  1
          |   |
          |   |
    2  h---------h'  3
    Return res, a non-invariant SU2kMPS with shape:
    [psi.shape[0],psi.shape[4],psi.shape[0],psi.shape[4]]
    heights:
    [psi.heights[0],psi.heights[4],psi.heights[0],psi.heights[4]]

    """
    heights_one_leg = psi_L.heights[0]
    shape_one_leg = psi_L.shape[0]

    if n_heights == 0:
        psi = psi_L.contract(psi_R)
        psip = psi
    elif n_heights == 1:
        psi = psi_L.contract(psi_R)
        if parity % 2 == 0:
            psip = act_Op(psi,i=0)
        else: # parity is odd
            psip = act_Op(psi,i=1)
    elif n_heights == 3:
        if parity % 2 == 0:
            # eg TL_gen acting of index equal to the second (middle)
            # height in psi:
            psi = psi_L.contract(psi_R)
            # actOp(psi) -> psip
            psip = act_Op(psi,i=1)
            # set shape and heights:
        else: # parity is odd
            # eg TL_gen acting of index equal to the third height in psi:
            # - psiL - psiR - psiL - psiR - 
            #    |      |      |      |
            #    |         Op         |
            #    |      |      |      |
            # - psiL - psiR - psiL - psiR -
            psi = psi_L.contract(psi_R)
            psi = psi.contract(psi_L)
            psi = psi.contract(psi_R)
            # actOp(psi) -> psip
            psip = act_Op(psi,i=2)
            # get E
    elif n_heights == 4:
        psi = psi_L.contract(psi_R)
        psi = psi.contract(psi_L)
        psi = psi.contract(psi_R)
        if parity % 2 == 0:
            # eg action with e1 \pm e2 or [e1,e2]
            # actOp(psi) -> psip
            psip = act_Op(psi,i=1)
        else:
            # actOp(psi) -> psip
            psip = act_Op(psi,i=2)
    elif n_heights == 5:
        if parity % 2 == 0:
            # eg action with [e1,e2]+[e2,e3] or S2
            # actOp(psi) -> psip
            psi = psi_L.contract(psi_R)
            psi = psi.contract(psi_L)
            psi = psi.contract(psi_R)
            psip = act_Op(psi,i=1)
        else:
            # odd parity, requires 6 sites
            psi = psi_L.contract(psi_R)
            psi = psi.contract(psi_L)
            psi = psi.contract(psi_R)
            psi = psi.contract(psi_L)
            psi = psi.contract(psi_R)
            # actOp(psi) -> psip
            psip = act_Op(psi,i=2)
    elif n_heights == 6:
        psi = psi_L.contract(psi_R)
        psi = psi.contract(psi_L)
        psi = psi.contract(psi_R)
        psi = psi.contract(psi_L)
        psi = psi.contract(psi_R)
        if parity % 2 == 0:
            psip = act_Op(psi,i=1)
        else:
            psip = act_Op(psi,i=2)
    else:
        print("In transfer operator: not implemented yet")

    # set shape and heights:
    E_sects = get_E_sects(psi, psip)
    E_shape = [shape_one_leg]*4
    E_heights = [heights_one_leg]*4
    res = SU2kMPS_file.SU2kMPS(E_shape, heights=E_heights, hmax=psi.hmax,
                               sects=E_sects, dtype=psi.dtype, invar=False)
    return res

def get_E_L(A,L,B,L_old):
    """Compute the left transfer operator of the SU2kMPS.

    """
    psi_L = A.contract(L.diag())
    if len(L_old.shape) == 1:
        psi_R = B.contract(L_old.diag().matrix_inv())
    elif len(L_old.shape) == 3:
        psi_R = B.contract(L_old.matrix_inv())
    else:
        print("In get_E_L: wrong shape of Lambda_old")
        return 0
    return transfer_operator(psi_L,psi_R)

def get_E_R(A,L,B,L_old):
    """Compute the right transfer operator of the SU2kMPS.
    
    """
    if len(L_old.shape) == 1:
        psi_L = L_old.diag().matrix_inv().contract(A)
    elif len(L_old.shape) == 3:
        psi_L = L_old.matrix_inv().contract(A)
    else:
        print("In get_E_R: wrong shape of Lambda_old")
        return 0
    psi_R = L.diag().contract(B)
    return transfer_operator(psi_L,psi_R)

def slice_dim_h(dim, height, h):
    """Returns the slice of indices corresponding 
    to height h and the dim corresponding to pos_h
    
    """
    pos_h = height.index(h)
    sh_sq_h = np.sum(np.array(dim[:pos_h],dtype=int)**2)
    sh_sq_h_p1 = np.sum(np.array(dim[:pos_h+1],dtype=int)**2)
    return slice(sh_sq_h,sh_sq_h_p1), dim[pos_h]

def get_E_mod(AA,AB):
    """Compute transfer operator of the SU2kMPS in AA,AB notation. Note,
    in this case we do not consider differently left/right transfer
    operators, just adjust later the update of lB.

    """
    return transfer_operator(AA,AB)

def contract_E_L(M, E_L):
    """Implements the contract of M with left indices of E_L.
    Like matvec_E_L but for SU2kMPS
    
    """
    assert(M.is_mat_like())
    assert(M.compatible_indices(E_L, 0, 0))
    ret=M.empty_like()
    for k,v in E_L.sects.items():
        h=k[0]
        hp=k[1]
        M_k = (h,1,h)
        M_block = M[M_k]
        ret_k = (hp,1,hp)
        ret_block = np.tensordot(M_block, v, axes=([0,2],[0,2])).transpose((1,0,2))
        if ret_k in ret.sects:
            ret[ret_k] += ret_block
        else:
            ret[ret_k] = ret_block
    return ret

def contract_E_R(M, E_R):
    """Implements the contract of M with right indices of E_R.
    
    """
    assert(M.is_mat_like())
    assert(M.compatible_indices(E_R, 0, 1))
    ret=M.empty_like()
    for k,v in E_R.sects.items():
        h=k[0]
        hp=k[1]
        M_k = (hp,1,hp)
        M_block = M[M_k]
        ret_k = (h,1,h)
        ret_block = np.tensordot(v, M_block, axes=([1,3],[0,2])).transpose((0,2,1))
        if ret_k in ret.sects:
            ret[ret_k] += ret_block
        else:
            ret[ret_k] = ret_block
    return ret

def trace_L_E_L(E_L):
    """Implements the trace over left indices of E_L
    
    """
    dims = E_L.shape[0]
    hs = E_L.heights[0]
    ret = SU2kMPS_file.SU2kMPS.eye(dims, hs=hs, hmax=E_L.hmax)
    ret = ret.empty_like()
    for k,v in E_L.sects.items():
        h=k[0]
        hp=k[1]
        ret_k = (hp,1,hp)
        ret_block = np.trace(v, axis1=0, axis2=2)
        if ret_k in ret.sects:
            ret[ret_k] += ret_block
        else:
            ret[ret_k] = ret_block
    return ret


def matvec_E_L(vec, E_L):
    """Implements the left multiplication of E_L with 
    the vector vec, an ndarray of shape (sum(E_L.shape[0]**2),)
    Assume heights are ordered, and for each h, compute
    cur_vec = vec[shape[..]]
    
    """
    hs = E_L.heights[0]
    dims = E_L.shape[0]
    ret = np.zeros(vec.shape)
    for h in hs:
        slc_h, dim_h = slice_dim_h(dims, hs, h)
        curv = vec[slc_h].reshape((dim_h,dim_h))
        for hp in hs:
            slc_hp, dim_hp= slice_dim_h(dims, hs, hp)
            # vec[slc_h] . E_L[h,hp] -> ret[slc_hp]
            block = E_L[h,hp,h,hp]
            tmp = np.tensordot(curv, block, axes=([0,1],[0,2]))
            ret[slc_hp] += tmp.reshape(dim_hp**2)
    return ret

def matvec_E_R(vec, E_R):
    """Implements the right multiplication of E_R with 
    the vector vec, an ndarray of shape (sum(E_R.shape[0]**2),)
    Assume heights are ordered.
    
    """
    hs = E_R.heights[0]
    dims = E_R.shape[0]
    ret = np.zeros(vec.shape)
    for h in hs:
        slc_h, dim_h = slice_dim_h(dims, hs, h)
        curv = vec[slc_h].reshape((dim_h,dim_h))
        for hp in hs:
            slc_hp, dim_hp= slice_dim_h(dims, hs, hp)
            # E_R[hp,h].vec[slc_h] -> ret[slc_hp]
            block = E_R[hp,h,hp,h]
            tmp = np.tensordot(curv, block, axes=([0,1],[1,3]))
            ret[slc_hp] += tmp.reshape(dim_hp**2)
    return ret

def check_norm_left(A):
    """Check that the A is left normalized

    """
    assert(len(A.shape)==3 and A.shape[1]==[1])
    # fuse left ind
    A_f = A.fuse(inds=(0,1))
    A_f_dag = A_f.transpose(p=(2,1,0)).conj()
    prod = A_f_dag.matrix_dot(A_f)
    myid = SU2kMPS_file.SU2kMPS.eye(A.shape[2],hs=A.heights[2],hmax=A.hmax)
    is_norm = prod.allclose(myid)
    if is_norm == False:
        print("In check_norm_left: ",prod-myid)
    return is_norm

def check_norm_right(B):
    """Check that the B is right normalized

    """
    assert(len(B.shape)==3 and B.shape[1]==[1])
    # fuse right ind
    B_f = B.fuse(inds=(2,1))
    B_f_dag = B_f.transpose(p=(2,1,0)).conj()
    prod = B_f.matrix_dot(B_f_dag)
    is_norm = prod.allclose(SU2kMPS_file.SU2kMPS.eye(B.shape[0],
                                                     hs=B.heights[0],hmax=B.hmax))
    return is_norm

def get_EEnt(A,Lambda,B,Lambda_old):
    """Compute the entanglement entropy from the matrices in the argument.

    """
    A1,A2,Lold_t,B1,B2=mixed_canonical_form(A,Lambda,B,Lambda_old)
    return get_EEent_mixed_repr(Lold_t)

def get_EEnt_mod(lA,lB,AA,AB):
    """Compute the entanglement entropy from the matrices in the argument.

    """
    A1,A2,Lt,B1,B2=mixed_canonical_form_mod(lA,lB,AA,AB)
    return get_EEent_mixed_repr(Lt)

def get_EEent_mixed_repr(L):
    """Computes Ent entropy of a state in mixed repr with central
    matrix L.

    """
    assert(L.is_mat_like())
    U, s, V, err = L.matrix_svd(remove_small=False)
    s=s.to_ndarray()
    # Check that sum s_a^2 = 1 and real
    if np.isreal(s).all() != True or np.isclose(sum(s**2),1.) == False:
        print ("ERROR: in get_EEent_mixed_repr, sing values not real or not normalized")
        return 0
    else:
        return -sum((s**2)*np.log(s**2))
    
def compute_corr_lengths(E_L, N):
    """Computes the first N correlation lengths associated to the left transfer
    operator my_E: -1/log(l_i), where l_i are the eigenvalues of my_E
    (left/right eigenvalues are the same so no worry about what
    action)

    """
    mv = lambda vec: matvec_E_L(vec, E_L)
    # Dimension
    dims = E_L.shape[0]
    D = np.sum(np.array(dims)**2)
    # Create linear operator od dim D x D
    E_L_lin_op = LinearOperator((D, D), matvec=mv, dtype=E_L.dtype)   
    # Compute the largest N eigenvalues of E_L.
    vals = eigs(E_L_lin_op, k=N, which='LM', return_eigenvectors=False)
    print("In compute_corr_lengths: diagonalization done, vals: ", vals)
    if np.all(np.imag(vals) == np.zeros(N)):
        vals=np.real(vals)
    else:
        warnings.warn("In compute_corr_lengths: vals not real")
    if np.allclose(max(vals), 1.) == False:
        print ("In compute_corr_lengths: ERROR: largest eig not 1")
        return 0
    vals = np.sort(vals)
    # Compute the correlation lenghts
    corr_lengths = -1./np.log(vals)

    return corr_lengths

def fidelity_mod(lB,AA):
    """Compute the fidelity as in 10083477, eq 342

    """
    # lB here is assumed to be a vector
    assert(len(lB.shape)==1)
    lB_mat = lB.diag()
    psi = lB_mat.contract(AA)
    # note: fuse right indices since then Lambda has to be contracted
    # with lB again
    psi = psi.fuse(inds=(2,1))
    # remove_small false since we do not want the dimension of S
    # to be automatically truncated because there are small values
    # otherwise not possible to take product with lB
    U,S,V,err,sqrtS = psi.matrix_svd(norm_S=False,remove_small=False)
    LambdaL = U.matrix_dot(S.diag())
    # lB^dagger = lB.conj
    #print(LambdaL.shape,lB_mat.shape)
    tmp = LambdaL.matrix_dot(lB_mat.conj())
    #print(tmp)
    U,S,V,err,sqrtS = tmp.matrix_svd(norm_S=False,remove_small=False)
    s_vec = S.to_ndarray()
    if s_vec.size > 1:
        return np.sum(s_vec)
    else: # at initial run, it might be trivial, just return 0
        return 0

def mixed_canonical_form(A,L,B,L_old,check=False):
    """Orthogonalize the left and right transfer matrices.
    Then compute the mixed canonical form, return A1,A2,new_L_old,B1,B2
    s.t the wf is
    A1.A2.A1.A2 ... new_L_old B1.B2.B1.B2 ...
    
    In the process normalize new_L_old.
    
    """
    X,eta1=compute_X(A,L,B,L_old)
    Y,eta2=compute_Y(A,L,B,L_old)
    if np.allclose(eta1, eta2) == False:
        print("Error: eta1 != eta2")
        return 0
    # compute A1, A2:
    L = L/np.sqrt(eta1)
    tmp_l = X.contract(A, erase_id=True)
    tmp_l = tmp_l.contract(L.diag(), erase_id=True)
    XLold = X.matrix_dot(L_old.diag())
    tmp_r = B.contract(XLold.matrix_inv())
    psi = tmp_l.contract(tmp_r)
    # do not normalize S to 1 
    A1, s, V, rel_err = psi.svd_2_sites(norm_S=False)
    A2 = s.diag().contract(V, erase_id=True)
    # compute B1, B2:
    LoldY = L_old.diag().matrix_dot(Y)
    tmp_l = LoldY.matrix_inv().contract(A, erase_id=True)
    tmp_l = tmp_l.contract(L.diag(), erase_id=True)
    tmp_r = B.contract(Y, erase_id=True)
    psi = tmp_l.contract(tmp_r)
    # do not normalize S to 1 
    U, s, B2, rel_err = psi.svd_2_sites(norm_S=False)
    B1 = U.contract(s.diag(), erase_id=True)

    if check:
        # check normalization:
        A_new = X.contract(A, erase_id=True)
        B_new = B.contract(Y, erase_id=True)
        L_new = L
        L_old_new = X.matrix_dot(L_old.diag())
        L_old_new = L_old_new.matrix_dot(Y)
        E_L = get_E_L(A_new,L_new,B_new,L_old_new)
        E_R = get_E_R(A_new,L_new,B_new,L_old_new)
        dims = E_L.shape[0]
        hs = E_L.heights[0]
        D = np.sum(np.array(dims)**2)
        my_id = SU2kMPS_file.SU2kMPS.eye(dims, hs=hs, hmax=E_L.hmax)
        my_id_v = np.zeros(D)
        for h in my_id.heights[0]:
            slc, d = slice_dim_h(dims, hs, h)
            my_id_v[slc] = my_id[h,1,h].reshape(d**2)
        is_norm_l = np.allclose(my_id_v,matvec_E_L(my_id_v,E_L))
        is_norm_r = np.allclose(my_id_v,matvec_E_R(my_id_v,E_R))
        print("In mixed_canonical_form: E_L,E_R normalized:",
              is_norm_l and is_norm_r)

    new_L_old = XLold.matrix_dot(Y)
    norm = new_L_old.transpose(p=(2,1,0)).conj().matrix_dot(new_L_old)
    norm = np.sqrt(norm.matrix_trace())
#    print("Norm =", norm)
    new_L_old /= norm
    
    return A1,A2,new_L_old,B1,B2

def mixed_canonical_form_mod(lA,lB,AA,AB,check=False):
    """Orthogonalize the left and right transfer matrices.
    Then compute the mixed canonical form, return A1,A2,lB_new,B1,B2
    s.t the wf is
    A1.A2.A1.A2 ... lB_new B1.B2.B1.B2 ...
    
    In the process normalize lB_new, so that the wave function is normalized.
    
    """
    lB_mat = lB.diag()
    pl=lB_mat.contract(AA)
    pr=AB.contract(lB_mat.matrix_inv())
    X,eta1=compute_X_mod(pl,pr,check=check)
    Xt = X.matrix_dot(lB_mat)
    Y,eta2=compute_Y_mod(AA,AB,check=check)
    if np.allclose(eta1, eta2) == False:
        print("Error: eta1 != eta2")
        return 0
    #
    # compute A1, A2:
    tmp_l = Xt.contract(AA, erase_id=True)
    tmp_l = tmp_l/np.sqrt(eta1)
    tmp_r = AB.contract(Xt.matrix_inv(), erase_id=True)
    psi = tmp_l.contract(tmp_r)
    # do not normalize S to 1 
    A1, s, V, rel_err = psi.svd_2_sites(norm_S=False)
    A2 = s.diag().contract(V, erase_id=True)
    # compute B1, B2:
    tmp_l = Y.matrix_inv().contract(AA, erase_id=True)
    tmp_l = tmp_l/np.sqrt(eta2)
    tmp_r = AB.contract(Y, erase_id=True)
    psi = tmp_l.contract(tmp_r)
    # do not normalize S to 1 
    U, s, B2, rel_err = psi.svd_2_sites(norm_S=False)
    B1 = U.contract(s.diag(), erase_id=True)

    if check:
        # check normalization:
        AA_new = Y.matrix_inv().contract(AA, erase_id=True)
        AB_new = AB.contract(Y, erase_id=True)
        E_R = get_E_mod(AA_new,AB_new)
        AA_new = Xt.contract(AA, erase_id=True)
        AB_new = AB.contract(Xt.matrix_inv(), erase_id=True)
        E_L = get_E_mod(AA_new,AB_new)
        dims = E_L.shape[0]
        hs = E_L.heights[0]
        my_id = SU2kMPS_file.SU2kMPS.eye(dims, hs=hs, hmax=E_L.hmax)
        ret=contract_E_L(my_id,E_L)
        is_norm_l = ret.allclose(eta1*my_id)
        ret=contract_E_R(my_id,E_R)
        is_norm_r = ret.allclose(eta2*my_id)
        print("In mixed_canonical_form: E_L,E_R normalized:",
              is_norm_l, is_norm_r)
        print("left norm:", check_norm_left(A1), check_norm_left(A2))
        print("right norm:", check_norm_right(B1), check_norm_right(B2))
    
    lB_new = Xt.matrix_dot(Y)
    norm = lB_new.transpose(p=(2,1,0)).conj().matrix_dot(lB_new)
    norm = np.sqrt(norm.matrix_trace())
#    print("Norm =", norm)
    lB_new /= norm
    
    return A1,A2,lB_new,B1,B2

def compute_X(A,L,B,L_old,check=False):
    """Computes X by diagoanalizing the left transfer matrix of the unit
    cell and compute the dominant eigenvector V_L, V_L = X^\dagger X
        
    """
    E_L = get_E_L(A,L,B,L_old)
    mv = lambda vec: matvec_E_L(vec, E_L)
    # Dimension
    dims = E_L.shape[0]
    D = np.sum(np.array(dims)**2)
    # Create linear operator od dim D x D
    E_L_lin_op = LinearOperator((D, D), matvec=mv, dtype=E_L.dtype)   
    # Compute the dominant eigenvectors of E_L.
    # Note, the transfer operator is Hermitian in general, so use eigs
    vals, vecs = eigs(E_L_lin_op, k=2, which='LM', return_eigenvectors=True)
#    print("In compute_X: diagonalization done, vals: ", vals)
    if np.all(np.imag(vecs) == np.zeros(vecs.shape)) and np.all(np.imag(vals) == np.zeros(2)):
        vecs=np.real(vecs)
        vals=np.real(vals)
    else:
        warnings.warn("In compute_X: vecs or vals not real")
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    eigv = vecs[:,max_ind]
    assert(np.allclose(linalg.norm(eigv),1.0))
    # check consistency:
#    print("check:",np.allclose(matvec_E_L(eigv, E_L), max_eig*eigv))
    dims = E_L.shape[0]
    hs = E_L.heights[0]
    V_L = SU2kMPS_file.SU2kMPS(shape=[dims,[1],dims], heights=[hs,[1],hs], hmax=E_L.hmax, invar=True)
    # assign blocks
    for h in hs:
        slc, d = slice_dim_h(dims, hs, h)
        V_L.sects[h,1,h] = eigv[slc].reshape((d,1,d))    
    # check hermiticity
    V_L_dag = V_L.transpose(p=(2,1,0)).conj()
    if V_L.allclose(V_L_dag) == False:
        print("ERROR: first In compute_X: eigv not hermitian. ")
        return 0
    # Now determine X_h from the eigendecomposition of V_L:
    S, U, rel_err = V_L.matrix_eig(hermitian=True)
    # and if V_L non-negative - as it should be. We have seen that
    # sometimes the global sign is wrong.  Clearly this can be
    # reabsorbed in the definition of V_L, so check if all negative,
    # just return -w
    w = S.to_ndarray()
    if np.all(w<=0):
        sign = -1.
    elif np.all(w>=0): 
        sign = 1.
    else: #means neither all >=0 nor all <=0, sign changes.
        warnings.warn("ERROR: In compute_X: V_L not >= 0, w ",w)
        return 0    
    # Finally compute V_L = X^dagger X:
    S = sign*S
    X = S.sqrt().diag().matrix_dot(U.conj().transpose(p=(2,1,0)))
    # Sanity check
    if check:
        tmp=X.transpose(p=(2,1,0)).conj().matrix_dot(X)
        assert(tmp.allclose(sign*V_L))

    return X, max_eig

def compute_X_mod(A,B,check=False):
    """Computes X by diagonalizing the left transfer matrix of the unit
    cell and compute the dominant eigenvector V_L, V_L = X^\dagger X
    Note: In principle, we could diagonalize the transfer operator
    based on AA.AB, however it turns out that the eigenvectors involve
    small number that cause problems when we have to invert X. Instead
    the good and stable object to work with is lB.AA.AB.lB.inv, as in
    the non modif case.  (Xt=X lB)

    """
    E_L = get_E_mod(A,B)
    mv = lambda vec: matvec_E_L(vec, E_L)
    # Dimension
    dims = E_L.shape[0]
    D = np.sum(np.array(dims)**2)
    # Create linear operator od dim D x D
    E_L_lin_op = LinearOperator((D, D), matvec=mv, dtype=E_L.dtype)   
    # Compute the dominant eigenvectors of E_L.
    # Note, the transfer operator is Hermitian in general, so use eigs
    vals, vecs = eigs(E_L_lin_op, k=2, which='LM', return_eigenvectors=True)
#    print("In compute_X: diagonalization done, vals: ", vals)
    if np.all(np.imag(vecs) == np.zeros(vecs.shape)) and np.all(np.imag(vals) == np.zeros(2)):
        vecs=np.real(vecs)
        vals=np.real(vals)
    else:
        warnings.warn("In compute_X_mod: vecs or vals not real")
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    eigv = vecs[:,max_ind]
    assert(np.allclose(linalg.norm(eigv),1.0))
    # check consistency:
    if check:
        print("check E_L*eigv=eta*eigv:",
              np.allclose(matvec_E_L(eigv, E_L), max_eig*eigv))
    dims = E_L.shape[0]
    hs = E_L.heights[0]
    V_L = SU2kMPS_file.SU2kMPS(shape=[dims,[1],dims], heights=[hs,[1],hs], hmax=E_L.hmax, invar=True)
    # assign blocks
    for h in hs:
        slc, d = slice_dim_h(dims, hs, h)
        V_L.sects[h,1,h] = eigv[slc].reshape((d,1,d))
    # check hermiticity
    V_L_dag = V_L.transpose(p=(2,1,0)).conj()
    if check:
        if V_L.allclose(V_L_dag) == False:
            print("ERROR: first In compute_X_mod: eigv not hermitian. ")
            return 0
        # check V_L * E_L = max_eig * V_L
        ret=contract_E_L(V_L,E_L)
        print("check V_L * E_L=eta*V_L:", ret.allclose(max_eig*V_L))
    # better numerical accuracy for hermiticity:
    V_L = (V_L + V_L_dag)/2
    # Now determine X_h from the eigendecomposition of V_L:
    S, U, rel_err = V_L.matrix_eig(hermitian=True)
    # and if V_L non-negative - as it should be. We have seen that
    # sometimes the global sign is wrong.  Clearly this can be
    # reabsorbed in the definition of V_L, so check if all negative,
    # just use abs(w), unless the sign oscillates which is an error
    tol = 1e-12 #tolerance for a number to be considered zero
    w = S.to_ndarray()
    if np.all(w > -tol):
        sign = 1.
    elif np.all(w < tol): 
        sign = -1.
    else: #means neither all >=0 nor all <=0, sign changes.
        print("ERROR: In compute_X: V_L not >= 0")
        print(w)
        sys.exit(1)
    # Finally compute V_L = X^dagger X:
    S = S.abs()
    X = S.sqrt().diag().matrix_dot(U.conj().transpose(p=(2,1,0)))
    # Sanity check
    if check:
        tmp=X.transpose(p=(2,1,0)).conj().matrix_dot(X)
        print("In compute_X_mod: XXdag=sign*VL", tmp.allclose(sign*V_L))
        
    return X, max_eig

def compute_Y(A,L,B,L_old,check=False):
    """Computes Y by diagoanalizing the right transfer matrix of the unit
    cell and compute the dominant eigenvector V_R, V_R = Y Y^dagger
        
    """
    E_R = get_E_R(A,L,B,L_old)
    mv = lambda vec: matvec_E_R(vec, E_R)
    # Dimension
    dims = E_R.shape[0]
    D = np.sum(np.array(dims)**2)
    # Create linear operator od dim D x D
    E_R_lin_op = LinearOperator((D, D), matvec=mv, dtype=E_R.dtype)   
    # Compute the dominant eigenvectors of E_R.
    # Note, the transfer operator is Hermitian in general, so use eigs
    vals, vecs = eigs(E_R_lin_op, k=2, which='LM', return_eigenvectors=True)
#    print("In compute_Y: diagonalization done, vals: ", vals)
    if np.all(np.imag(vecs) == np.zeros(vecs.shape)) and np.all(np.imag(vals) == np.zeros(2)):
        vecs=np.real(vecs)
        vals=np.real(vals)
    else:
        warnings.warn("In compute_Y: vecs or vals not real")
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    eigv = vecs[:,max_ind]
    assert(np.allclose(linalg.norm(eigv),1.0))
    # check consistency:
#    print("check:",np.allclose(matvec_E_R(eigv, E_R), max_eig*eigv))
    dims = E_R.shape[0]
    hs = E_R.heights[0]
    V_R = SU2kMPS_file.SU2kMPS(shape=[dims,[1],dims], heights=[hs,[1],hs], hmax=E_R.hmax, invar=True)
    # assign blocks
    for h in hs:
        slc, d = slice_dim_h(dims, hs, h)
        V_R.sects[h,1,h] = eigv[slc].reshape((d,1,d))    
    # check hermiticity
    V_R_dag = V_R.transpose(p=(2,1,0)).conj()
    if V_R.allclose(V_R_dag) == False:
        print("ERROR: first In compute_Y: eigv not hermitian.")
        return 0
    # Now determine Y_h from the eigendecomposition of V_R:
    S, U, rel_err = V_R.matrix_eig(hermitian=True)
    # and if V_R non-negative - as it should be. We have seen that
    # sometimes the global sign is wrong.  Clearly this can be
    # reabsorbed in the definition of V_R, so check if all negative,
    # just return -w
    w = S.to_ndarray()
    if np.all(w<=0):
        sign = -1.
    elif np.all(w>=0): 
        sign = 1.
    else: #means neither all >=0 nor all <=0, sign changes.
        warnings.warn("ERROR: In compute_Y: V_R not >= 0, w ",w)
        return 0    
    # Finally compute V_R = Y Y^dagger:
    S = sign*S
    Y = U.matrix_dot(S.sqrt().diag())
    # Sanity check
    if check:
        tmp=Y.matrix_dot(Y.transpose(p=(2,1,0)).conj())
        assert(tmp.allclose(sign*V_R))

    return Y, max_eig

def compute_Y_mod(AA,AB,check=False):
    """Computes Y by diagonalizing the right transfer matrix of the unit
    cell and compute the dominant eigenvector V_R, V_R = Y Y^dagger
        
    """
    E_R = get_E_mod(AA,AB)
    mv = lambda vec: matvec_E_R(vec, E_R)
    # Dimension
    dims = E_R.shape[0]
    D = np.sum(np.array(dims)**2)
    # Create linear operator od dim D x D
    E_R_lin_op = LinearOperator((D, D), matvec=mv, dtype=E_R.dtype)   
    # Compute the dominant eigenvectors of E_R.
    # Note, the transfer operator is Hermitian in general, so use eigs
    vals, vecs = eigs(E_R_lin_op, k=2, which='LM', return_eigenvectors=True)
#    print("In compute_Y: diagonalization done, vals: ", vals)
    if np.all(np.imag(vecs) == np.zeros(vecs.shape)) and np.all(np.imag(vals) == np.zeros(2)):
        vecs=np.real(vecs)
        vals=np.real(vals)
    else:
        warnings.warn("In compute_Y: vecs or vals not real")
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    eigv = vecs[:,max_ind]
    assert(np.allclose(linalg.norm(eigv),1.0))
    # check consistency:
#    print("check:",np.allclose(matvec_E_R(eigv, E_R), max_eig*eigv))
    dims = E_R.shape[0]
    hs = E_R.heights[0]
    V_R = SU2kMPS_file.SU2kMPS(shape=[dims,[1],dims], heights=[hs,[1],hs], hmax=E_R.hmax, invar=True)
    # assign blocks
    for h in hs:
        slc, d = slice_dim_h(dims, hs, h)
        V_R.sects[h,1,h] = eigv[slc].reshape((d,1,d))    
    # check hermiticity
    V_R_dag = V_R.transpose(p=(2,1,0)).conj()
    if V_R.allclose(V_R_dag) == False:
        print("ERROR: first In compute_Y: eigv not hermitian.")
        return 0
    # Now determine Y_h from the eigendecomposition of V_R:
    S, U, rel_err = V_R.matrix_eig(hermitian=True)
    # and if V_R non-negative - as it should be. We have seen that
    # sometimes the global sign is wrong.  Clearly this can be
    # reabsorbed in the definition of V_R, so check if all negative,
    # just return -w
    w = S.to_ndarray()
    if np.all(w<=0):
        sign = -1.
    elif np.all(w>=0): 
        sign = 1.
    else: #means neither all >=0 nor all <=0, sign changes.
        warnings.warn("ERROR: In compute_Y: V_R not >= 0, w ",w)
        return 0    
    # Finally compute V_R = Y Y^dagger:
    S = sign*S
    Y = U.matrix_dot(S.sqrt().diag())
    # Sanity check
    if check:    
        tmp=Y.matrix_dot(Y.transpose(p=(2,1,0)).conj())
        assert(tmp.allclose(sign*V_R))

    return Y, max_eig

#######################
#    correlators      #
#######################

def multiply_Lold_and_trace(E, L_old):
    """Multiply the transfer operator E by L_old and trace to produce a
    number

    """
    assert(L_old.is_mat_like())
    res = 0
    for k,v in E.sects.items():
        h = k[0]
        hp = k[1]
        vp = np.trace(v,axis1=0,axis2=2)
        m = L_old[hp,1,hp]
        shp = m.shape[0]
        m = m.reshape((shp,shp))
        vp = np.tensordot(vp,m,axes=(1,0))
        res += np.tensordot(np.conj(m),vp,axes=([0,1],[0,1]))
    return res

def multiply_transfer_operators(E1, E2):
    """Multiply the transfer operator E1 by E2 contracting right indices
    of E1 with left of E2.

    """
    assert(E1.compatible_indices(E2, 1, 0) and E1.compatible_indices(E2, 3, 2))
    res = E1.empty_like()
    for k1,v1 in E1.sects.items():
        h = k1[0]
        hp = k1[1]
        for hpp in E2.heights[1]:
            # res.sects[h,hpp,h,hpp]=
            # sum(E1.sects[h,hp,h,hp].E2.sects[hp,hpp,hp,hpp],hp)
            k2 = (hp,hpp,hp,hpp)
            if k2 in E2.sects:
                v2 = E2[k2]
                new_k = (h,hpp,h,hpp)
                tmp = np.tensordot(v1,v2,axes=([1,3],[0,2]))
                tmp = np.transpose(tmp, (0,2,1,3))
                # this implements the sum over hp
                if new_k in res.sects:
                    res[new_k] += tmp
                else:
                    res[new_k] = tmp
    return res

def one_point_function(A1, A2, L_old, n_heights=0, parity=0, act_Op=act_id):
    """Computes <psi|op|psi>. See transfer_operator for explanation of
    arguments.

    """
    # No matter what Op is considered, E has always the same shape.
    # One point function, just contract with L_old and trace
    E = transfer_operator(A1, A2, n_heights=n_heights, parity=parity,
                          act_Op=act_Op)
    return multiply_Lold_and_trace(E, L_old)
 
def two_point_function(A1, A2, L_old, x, shift = 0, n_heights_1=0, n_heights_2=0, 
                       act_Op_1=act_id, act_Op_2=act_id):
    """Computes <psi|Op_1(1) Op_2(x)|psi> if shift % 2 = 0 and
    <psi|Op_1(2) Op_2(x+1)|psi> if shift % 2 = 1.
    
    See transfer_operator for explanation of arguments. 1 and x stand
    for the first site on which the operators act.

    """
    E_1 = transfer_operator(A1, A2, n_heights=n_heights_1, parity=shift,
                            act_Op=act_Op_1)
    E_2 = transfer_operator(A1, A2, n_heights=n_heights_2, parity=x+shift,
                            act_Op=act_Op_2)
    E = transfer_operator(A1, A2, n_heights=0, parity=0, act_Op=act_id)
    # l = number of id transf operators from E_1 to E_2

    print("In two_point_function: TODO: Double check offset and shift!!!!!!")
    
    if n_heights_1 == 1:
        if shift == 0 or (shift == 1 and x % 2 == 0):
            offset = 1
        else: # shift=1 and x odd
            offset = 0
    elif n_heights_1 == 3 and shift == 0:
        offest = 1
    elif (n_heights_1 == 4 and shift == 0) or (n_heights_1 == 5 and shift == 0):
        offset = 2
    elif n_heights_1 == 6 and shift == 0:
        offset = 3
    else:
        print("In two_point_function: not implemented yet")
    if x % 2 == 0:
        l = int(x/2)-offset
    else:
        l = int((x-1)/2)-offset
    curE = E_1

    print('shift,x,offset',shift,x,offset)
    for n in range(l):
        curE = multiply_transfer_operators(curE, E)
    curE = multiply_transfer_operators(curE, E_2)
    
    return multiply_Lold_and_trace(curE, L_old)


def two_point_function_range_x(A1, A2, L_old, x1, xN, n_heights_1=0, n_heights_2=0, 
                                         act_Op_1=act_id, act_Op_2=act_id):
    """Compute < Op1(x1) Op2(x) > for all x1 < x <= xN

    """
    # first transfer operator
    E_1 = transfer_operator(A1, A2, n_heights=n_heights_1, parity=x1,
                            act_Op=act_Op_1)
    # define second transfer operator, one for even x, one for odd x:
    E_2_ev = transfer_operator(A1, A2, n_heights=n_heights_2, parity=0,
                               act_Op=act_Op_2)
    E_2_odd = transfer_operator(A1, A2, n_heights=n_heights_2, parity=1,
                                act_Op=act_Op_2)
    # transfer operator with identity
    E = transfer_operator(A1, A2, n_heights=0, parity=0, act_Op=act_id)

    dist = []
    corr = []
    curE = E_1
    if n_heights_1 == 1 and n_heights_2 == 1:
        if x1 % 2 == 0:
            start = x1 + 2
        else:
            start = x1 + 1
        # note: start is always even
        for x in range(start,xN+1):
            i = x - x1
            if x % 2 == 0:
                resE = multiply_transfer_operators(curE, E_2_ev)
            else:
                resE = multiply_transfer_operators(curE, E_2_odd)
            my_corr = multiply_Lold_and_trace(resE, L_old)
            dist.append(i)
            corr.append(my_corr)
            # every two x's update curE
            if x % 2 == 1 and x != xN:
                curE = multiply_transfer_operators(curE, E)
#            print('x,i,my_corr',x,i,my_corr)
    
    return dist, corr
