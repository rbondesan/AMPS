"""EE_vs_D: 
compute the dependence of the entanglement entropy
vs log(xi), where xi is the correlation length,
related to the second largest eigenvalue of the MPS
transfer matrix, at a given D. 

It calls routines defined in package modules/
"""

import sys
import numpy as np
import pickle
from modules import functions,SU2k_data

def main():
    # Lists
    D_list = range(10,70,10) #list(range(10,52,2))+[60,70,80] 
    logxi_list = []
    EE_list = []
    log_corr_lens_list = []
    # Model global vars
    pars = {}
    pars['hmax'] = 4
    pars['dt'] = 1e-01
    pars['eps'] = 1e-12
    # boundary condition
    bc = 1
    pars['height0'] = bc
    pars['heightL'] = bc
    # number of eigenvalues of E to be computed
    n_corr_lens = 5

    for cur_D in D_list:
        print("In EE_vs_D: cur_D ", cur_D)
        # read data
        pars['chi'] = cur_D
        lA,lB,AA,AB,A1,A2,Lold_t,B1,B2=functions.read_from_file_mod(pars)

        # The wave function is
        # A1.A2.A1.A2 ... A1.A2 Lold_t B1.B2.B1.B2 ...
        # A1,A2,Lold_t,B1,B2=functions.mixed_canonical_form(A,Lambda,B,
        #                                         Lambda_old,check=True)
        # and the left transfer matrix is A1.A2, the right one B1.B2
        # Compute first 16 corr lengths, EE, bond energy 
        print("Check l_norm",functions.check_norm_left(A1),
              functions.check_norm_left(A2))
        print("Check r_norm",functions.check_norm_right(B1),
              functions.check_norm_right(B2))
        E_L = functions.transfer_operator(A1,A2)
        cur_corr_lens = functions.compute_corr_lengths(E_L,n_corr_lens)
        cur_corr_lens = np.real(cur_corr_lens[0:-1])#up to the last which is infty.
        cur_xi = max(cur_corr_lens) 
        print("log corr lens ", np.log(cur_corr_lens))
        print("* cur_xi ", cur_xi)
        # Compute EE
        cur_EE = functions.get_EEent_mixed_repr(Lold_t)
        print("log(xi) vs EE ", np.log(cur_xi), cur_EE)
        # And append to lists
        log_corr_lens_list.append(np.log(cur_corr_lens))
        logxi_list.append(np.log(cur_xi))
        EE_list.append(cur_EE)
    
        # Print the bond energy as a check
        e_infty = SU2k_data.e_infty(pars['hmax'])
        En_0 = functions.one_point_function(A1,A2,Lold_t,n_heights=3,parity=0,
                                            act_Op=functions.act_TL_gen)
        En_1 = functions.one_point_function(A1,A2,Lold_t,n_heights=3,parity=1,
                                            act_Op=functions.act_TL_gen)
        print("Bond energy: ", np.mean([En_0,En_1]), ". Exact: ", e_infty)
        print("* Delta bond energy: ", np.mean([En_0,En_1]) - e_infty)
        # Go to next D

    # Print to a handy file to analyze
    out_file_name = 'results/SU2k_itebd_logxi_vs_EE_hmax_'+str(pars['hmax'])+\
        '_a_'+str(pars['height0'])+'_b_'+str(pars['heightL'])+'_'
    if len(D_list) == 1:
        out_file_name += 'chi_'+str(cur_D)+'_'
    else:
        out_file_name += 'chi_'+str(D_list[0])+'_'+str(D_list[-1])+'_'
    out_file_name += 'dt_'+str(pars['dt'])+'_'
    out_file_name += 'eps_'+'{:.0e}'.format(pars['eps'])
    f = open(out_file_name, 'w')
    n = 0
    for cur_D in D_list:
        f.write(str(cur_D)+" "+str(logxi_list[n])+" "+str(EE_list[n])+"\n")
        n = n + 1
    f.close()
    print('saved to file ', out_file_name)
    # 
    out_file_name = 'results/SU2k_itebd_chi_vs_log_corr_lens_hmax_'+str(pars['hmax'])+\
        '_a_'+str(pars['height0'])+'_b_'+str(pars['heightL'])+'_'
    if len(D_list) == 1:
        out_file_name += 'chi_'+str(cur_D)+'_'
    else:
        out_file_name += 'chi_'+str(D_list[0])+'_'+str(D_list[-1])+'_'
    out_file_name += 'dt_'+str(pars['dt'])+'_'
    out_file_name += 'eps_'+'{:.0e}'.format(pars['eps'])
    f = open(out_file_name, 'w')
    n = 0
    for cur_D in D_list:
        f.write(str(cur_D))
        for x in log_corr_lens_list[n]:
            f.write(" "+str(x))
        f.write("\n")
        n = n + 1
    f.close()
    print('saved to file ', out_file_name)

    print('End of the program, exiting.')

# Executed if called as a script
if __name__ == '__main__':
    main()

