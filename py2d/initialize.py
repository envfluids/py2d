import numpy as np
import numpy as nnp
import jax.numpy as np
      
def initialize_random(NX, Kx, Ky):
    # -------------- Initialization using pertubration --------------
    kp = 10.0
    A  = 4*np.power(kp,(-5))/(3*np.pi)
    absK = np.sqrt(Kx*Kx+Ky*Ky)
    Ek = A*np.power(absK,4)*np.exp(-np.power(absK/kp,2))
    coef1 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
    coef2 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2

    perturb = np.zeros([NX,NX])
    perturb[0:NX//2+1, 0:NX//2+1] = coef1[0:NX//2+1, 0:NX//2+1]+coef2[0:NX//2+1, 0:NX//2+1]
    perturb[NX//2+1:, 0:NX//2+1] = coef2[NX//2-1:0:-1, 0:NX//2+1] - coef1[NX//2-1:0:-1, 0:NX//2+1]
    perturb[0:NX//2+1, NX//2+1:] = coef1[0:NX//2+1, NX//2-1:0:-1] - coef2[0:NX//2+1, NX//2-1:0:-1]
    perturb[NX//2+1:, NX//2+1:] = -(coef1[NX//2-1:0:-1, NX//2-1:0:-1] + coef2[NX//2-1:0:-1, NX//2-1:0:-1])
    perturb = np.exp(1j*perturb)

    w1_hat = np.sqrt(absK/np.pi*Ek)*perturb*np.power(NX,2)

    psi_hat = -w1_hat*invKsq
    psiPrevious_hat = psi_hat.astype(np.complex128)
    psiCurrent_hat = psi_hat.astype(np.complex128)
    
    return w1_hat, psi_hat, psiPrevious_hat, psiCurrent_hat
    
    
#def initialize_resume(NX, Kx, Ky):
