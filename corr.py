"""Compute correlation functions.

In RSOS models, one takes the modified trace, which implies that a
correlation function is:

< Op > = \sum_{a=1}^hmax <gs_a| Op |gs_a> S_a^2,

where |gs_a> is the ground state with boundary conditions a,a.

"""

import sys,argparse
import numpy as np
import modules.functions as funcs
import modules.SU2k_data as SU2k_data

def main(D):
    # set global parameters
    x1 = 1 #=0 or 1
    xN = 100
    hmax = 4
    dt = 1e-04
    eps = 1e-6
    # 
    myOp1=funcs.act_F1
    myOp1_name = 'F1'
    my_n_heights1=5
    myOp2=funcs.act_F1
    myOp2_name = 'F1'
    my_n_heights2=5
    # and pars
    pars = {}
    pars['chi'] = D
    pars['hmax'] = hmax
    pars['dt'] = dt
    pars['eps'] = eps

    # loop over boundary conditions a.
    # add the result to corr with the right factor S_a^2
    if my_n_heights1 == 1:
        N = xN-1
    elif my_n_heights1 == 4:
        N = xN-3
    elif my_n_heights1 == 5:
        N = xN-3
    corr = np.zeros(N)
    Vev_Op1 = 0
    Vev_Op2_ev = 0
    Vev_Op2_odd = 0
    Z = 0
    for a in range(1,hmax+1):
        # open file
        pars['height0'] = a
        pars['heightL'] = a
        lA,lB,AA,AB,A1,A2,Lold_t,B1,B2=funcs.read_from_file_mod(pars)
        print(A1.heights,A2.heights,Lold_t.heights)
        Sasq = SU2k_data.qpsi(hmax,a)**2
        # Partition function not 1, so compute it:
        Z += Sasq
        # one point functions to be subtracted
        tmp1 = Sasq * funcs.one_point_function(A1,A2,Lold_t,
                                               n_heights=my_n_heights1,
                                               parity=x1,act_Op=myOp1)
        tmp2 = Sasq * funcs.one_point_function(A1,A2,Lold_t,
                                               n_heights=my_n_heights2,
                                               parity=0,act_Op=myOp2)
        tmp3 = Sasq * funcs.one_point_function(A1,A2,Lold_t,
                                               n_heights=my_n_heights2,
                                               parity=1,act_Op=myOp2)
        Vev_Op1 += tmp1
        Vev_Op2_ev += tmp2
        Vev_Op2_odd += tmp3
        print("a",a,"<Op>_a",tmp1,tmp2,tmp3)
        cur_dist, cur_corr = funcs.two_point_function_range_x(A1,A2,Lold_t, 
                                                x1, xN, 
                                                n_heights_1=my_n_heights1, 
                                                n_heights_2=my_n_heights2, 
                                                act_Op_1=myOp1, 
                                                act_Op_2=myOp2)
        corr += np.array(cur_corr) * Sasq
    dist = cur_dist
    # normalize
    corr /= Z 
    Vev_Op1 /= Z
    Vev_Op2_ev /= Z
    Vev_Op2_odd /= Z
    # subtract 1 pf:    
    print("In corr_new: <Op1>",Vev_Op1)
    print("In corr_new: <Op2>",Vev_Op2_ev,Vev_Op2_odd)        
    if my_n_heights1 == 1 and my_n_heights2 == 1:
        # subtracted 1 pf. start from x=2 which is even
        corr[::2] -= Vev_Op1 * Vev_Op2_ev
        corr[1::2] -= Vev_Op1 * Vev_Op2_odd
    elif my_n_heights1 == 4 and my_n_heights2 == 4:
        # subtracted 1 pf. start from x=5 which is odd, but 
        # according to transfer operator, uses even parity E.
        corr[::2] -= Vev_Op1 * Vev_Op2_ev
        corr[1::2] -= Vev_Op1 * Vev_Op2_odd
    elif my_n_heights1 == 5 and my_n_heights2 == 5:
        # subtracted 1 pf. start from x=5 which is odd, but 
        # according to transfer operator, uses even parity E.
        corr[::2] -= Vev_Op1 * Vev_Op2_ev
        corr[1::2] -= Vev_Op1 * Vev_Op2_odd
    else:
        print('not implemented, exit')
        sys.exit(1)

    # save
    out_file_name = 'results/SU2k_itebd_corr_'+myOp1_name+'_'+myOp2_name+'_'
    out_file_name += 'i_'+str(x1)+'_'+str(xN)+'_hmax_'+str(pars['hmax'])+'_'
    out_file_name += 'chi_'+str(pars['chi'])+'_'
    out_file_name += 'dt_'+str(pars['dt'])+'_'
    out_file_name += 'eps_'+'{:.0e}'.format(pars['eps'])
    f = open(out_file_name, 'w')
    for a,b in zip(dist,corr):
        f.write(str(a)+" "+str(b))
        f.write("\n")
    f.close()
    print('saved to file ', out_file_name)
            
# Executed if called as a script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SU2k_iTEBD')
    parser.add_argument('-D', '--chi', type=int, help='max bond dimension')
    args = parser.parse_args()
    # store the parameters in dictionary pars
    D = vars(args)['chi']

    main(D)
