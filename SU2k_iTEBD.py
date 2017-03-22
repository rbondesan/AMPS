"""Ayonic iTEBD

"""

import argparse
from modules.functions import init_itebd, init_itebd_mod, recursion, recursion_mod, save_to_file, save_to_file_mod
    
def main(pars, max_it, save):
    """Performs the recursion to compute the cell matrices
    A,Lambda,B,Lambda_old and returns them

    max_it = max number of iterations, even number since parity of n =
    current iteration is that of site where U acts
    
    pars has the following keys: hmax,chi,dt,eps

    Example of global variables:
    hmax = 4
    chi = 20 # max schmidt rank
    delta_t = 0.01 # delta t in TEBD algorithm, U = exp(-delta_t h)
    eps = 10.**(-6) accuracy for convergence

    """    
    lA,lB,AA,AB = init_itebd_mod(pars)
    lA,lB,AA,AB,A1,A2,Lt,B1,B2 = recursion_mod(lA,lB,AA,AB,max_it,pars,save)
    
    if save:
        save_to_file_mod(lA,lB,AA,AB,A1,A2,Lt,B1,B2,pars)
        # do not return anything
    else:
        return lA,lB,AA,AB

# Executed if called as a script
if __name__ == '__main__':
    # Get from command line
    parser = argparse.ArgumentParser(description='SU2k_iTEBD')
    parser.add_argument('-p', '--hmax', type=int, help='max height')
    parser.add_argument('-D', '--chi', type=int, help='max bond dimension')
    parser.add_argument('-N', '--maxit', type=int, help='max iteration',
                        default=10**6)
    parser.add_argument('-t', '--deltat', type=float, help='delta_t')
    parser.add_argument('-e', '--eps', type=float, help='epsilon',
                        default=10.**(-6))
    parser.add_argument('-s', '--savetofile', type=bool, 
                        help='wether to save the matrices to file',
                        default=True)
    parser.add_argument('-a', '--height0', type=int, help='left bc',
                        default=1)
    parser.add_argument('-b', '--heightL', type=int, help='right bc',
                        default=1)
    args = parser.parse_args()
    save = vars(args)['savetofile']
    max_it = vars(args)['maxit']
    # store the parameters in dictionary pars
    pars = {}
    pars['hmax'] = vars(args)['hmax']
    pars['chi'] = vars(args)['chi']
    pars['dt'] = vars(args)['deltat']
    pars['eps'] = vars(args)['eps']
    pars['height0'] = vars(args)['height0']
    pars['heightL'] = vars(args)['heightL']

    main(pars, max_it, save)
