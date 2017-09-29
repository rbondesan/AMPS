"""
auxiliary file for SU2 level k:

created by RB on 25/01/17. 

"""
import numpy as np
from scipy import integrate

def qdim(hmax,h):
    """returns the quantum dimension [h]_q 

    """
    return np.sin(np.pi/(hmax+1) * h)/np.sin(np.pi/(hmax+1))

def Fmat(a,b,c,d,e,f,hmax):
    """returns the F matrix

    """
    if f == 1 and d == a and b == c:
        return np.sqrt(qdim(hmax,e) / qdim(hmax,a) / qdim(hmax,b))
    elif e == 1 and d == a and b == c:
        return np.sqrt(qdim(hmax,f) / qdim(hmax,a) / qdim(hmax,b))
    else:
        raise ValueError('In SU2k_data.Fmat: Not implemented')

def qpsi(hmax,h):
    """psi: 1/2 numerator of quantum number [h]_q

    """
    return np.sin(np.pi/(hmax+1) * h)

def qV(hmax,m,h):
    """psi: 1/2 numerator of quantum number [h]_q

    """
    return np.sin(np.pi/(hmax+1) * m * h)

def order_par_weight(hmax,m,h):
    """Order parameter Weight

    """
    return qV(hmax,m,h)/qpsi(hmax,h)

def TL_weight(hmax,him1,hi,hip):
    """Temperley-Lieb Weight

    """
    ret = np.sqrt(qpsi(hmax,hi) * qpsi(hmax, hip))/qpsi(hmax, him1)
    return ret
    
def fusion_range(hmax,h1,h2):
    """Return a tuple of fusion channels of a x b in SU2 level
    hmax-1

    """
    # the final + 1 is because the values go till min(...) included. Step is 2
    return range(abs(h1-h2)+1,min(h1+h2-1,2*hmax+1-h1-h2) + 1,2)

def Nmat_el(hmax,a,b,c):
    if a in fusion_range(hmax,b,c):
        return 1
    else:
        return 0

def e_infty(p):
    """Return <gs|e_i|gs>

    """
    gamma = np.pi/(p+1)
    Delta = np.cos(gamma)
    f = lambda x,g : 1/np.cosh(np.pi*x)\
        /(np.cosh(2*g*x) - np.cos(g))
    E0Al = integrate.quad(f, 0, np.inf, args=(gamma,))[0]
    E0Al = 1/2.*np.cos(gamma) - 2*np.sin(gamma)*np.sin(gamma)*E0Al
    # E0Al is the energy of Eq. 2.49
    # http://snoelieputruiq.com/doc/Publ_26_Surface_exponents.pdf
    E0 = E0Al - Delta/2.
    Einfty = -E0
    return Einfty

# def Nmat(hmax,a):
#     """Returns the fusion matrix N corresponding to anyon a in SU2 level
#     hmax-1

#     """
#     if hmax == 3:
#         if a == 1:
#             NN = np.eye(3, dtype=int)
#         elif a == 2:
#             NN = np.array([[0,1,0],
#                            [1,0,1],
#                            [0,1,0]])
#         elif a == 3:
#             NN = np.array([[0,0,1],
#                            [0,1,0],
#                            [1,0,1]])
#         else:
#             print("In Nmat: index a =",a,"is not valid")
#         return NN
#     elif hmax == 4:
#         if a == 1:
#             NN = np.eye(4, dtype=int)
#         elif a == 2:
#             NN = np.array([[0,1,0,0],
#                            [1,0,1,0],
#                            [0,1,0,1],
#                            [0,0,1,0]])
#         elif a == 3:
#             NN = np.array([[0,0,1,0],
#                            [0,1,0,1],
#                            [1,0,1,0],
#                            [0,1,0,1]])
#         elif a == 4:
#             NN = np.array([[0,0,0,1],
#                            [0,0,1,0],
#                            [0,1,0,1],
#                            [1,0,1,0]])
#         else:
#             print("In Nmat: index a =",a,"is not valid. Return 0")
#             return 0
#         return NN
#     elif hmax == 5:
#         if a == 1:
#             NN = np.eye(, dtype=int)
#         elif a == 2:
#             NN = np.array([[0,1,0,0],
#                            [1,0,1,0],
#                            [0,1,0,1],
#                            [0,0,1,0]])
#         elif a == 3:
#             NN = np.array([[0,0,1,0],
#                            [0,1,0,1],
#                            [1,0,1,0],
#                            [0,1,0,1]])
#         elif a == 4:
#             NN = np.array([[0,0,0,1],
#                            [0,0,1,0],
#                            [0,1,0,1],
#                            [1,0,1,0]])
#         else:
#             print("In Nmat: index a =",a,"is not valid. Return 0")
#             return 0
#         return NN
#     else:
#         print("In fusion_matrix: hmax", hmax, " not implemented yet. Return 0")
#         return 0
