import jax.numpy as np
import numpy as nnp

from py2d.eddy_viscosity_models import eddy_viscosity_smag, characteristic_strain_rate_smag, coefficient_dsmag_PsiOmega
from py2d.eddy_viscosity_models import eddy_viscosity_leith, characteristic_omega_leith, coefficient_dleith_PsiOmega
from py2d.gradient_model import PiOmegaGM2_2DFHIT, PiOmegaGM4_2DFHIT, PiOmegaGM6_2DFHIT
from py2d.eddy_viscosity_models import characteristic_omega_leith, coefficient_dleithlocal_PsiOmega
# from py2d.uv2tau_CNN import evaluate_model, init_model
from py2d.eddy_viscosity_models import Tau_eddy_viscosity
from py2d.convert import Tau2PiOmega_2DFHIT

class SGSModel:

    def __init__(self, Kx, Ky, Ksq, Delta, method = 'NoSGS', C_MODEL=0):
        self.set_method(method)
        # Constants
        self.Kx = Kx
        self.Ky = Ky
        self.Ksq = Ksq
        self.Delta = Delta
        self.C_MODEL = C_MODEL
        # States
        self.Psi_hat, self.PiOmega_hat, self.eddy_viscosity, self.Cl, self.Cs = 0, 0, 0, None, None

    def set_method(self, method):
        if method == 'NoSGS':
            self.calculate = self.no_sgs_method
        elif method == 'SMAG':
            self.calculate = self.smag_method
        elif method == 'DSMAG':
            self.calculate = self.dsmag_method
        elif method == 'LEITH':
            self.calculate = self.leith_method
        elif method == 'DLEITH':
            self.calculate = self.dleith_method
        elif method == 'DLEITH_tau_Local':
            self.calculate = self.dleithlocal_method
            self.localflag='from_tau'
        elif method == 'DLEITH_sigma_Local':
            self.calculate = self.dleithlocal_method
            self.localflag='from_sigma'
        elif method == 'PiOmegaGM2':
            self.calculate = self.PiOmegaGM2_method
        elif method == 'PiOmegaGM4':
            self.calculate = self.PiOmegaGM4_method
        elif method == 'PiOmegaGM6':
            self.calculate = self.PiOmegaGM6_method
        elif method == 'CNN':
            self.calculate = self.cnn_method
        elif method == 'GAN':
            self.calculate = self.gan_method
        else:
            raise ValueError(f"Invalid method: {method}")

    def __expand_self__(self):
        Kx = self.Kx
        Ky = self.Ky
        Ksq = self.Ksq
        Delta = self.Delta
        C_MODEL = self.C_MODEL
        return Kx, Ky, Ksq, Delta, C_MODEL

    def update_state(self, Psi_hat, Omega_hat, U_hat, V_hat):
        self.Psi_hat, self.Omega_hat = Psi_hat, Omega_hat
        self.U_hat, self.V_hat = U_hat, V_hat
        return None

    def no_sgs_method(self):
        PiOmega_hat = 0.0
        eddy_viscosity = 0.0
        return PiOmega_hat, eddy_viscosity


    def smag_method(self):#, Psi_hat, Cs, Delta):
        Kx, Ky, Ksq, Delta, Cs = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
        eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cs
        return PiOmega_hat, eddy_viscosity, Cs


    def leith_method(self):#, Omega_hat, Kx, Ky, Cl, Delta):
        Kx, Ky, Ksq, Delta, Cl = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl
        return PiOmega_hat, eddy_viscosity, Cl


    def dsmag_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
        c_dynamic = coefficient_dsmag_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
        Cs = np.sqrt(c_dynamic)
        eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cs
        return PiOmega_hat, eddy_viscosity, Cs

    def dleith_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        c_dynamic = coefficient_dleith_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        Cl = c_dynamic ** (1/3)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl

        return PiOmega_hat, eddy_viscosity, Cl

    def dleithlocal_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        #
        c_dynamic = coefficient_dleithlocal_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        Cl = c_dynamic ** (1/3)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        if self.localflag=='from_sigma':
            # Calculate the PI term for local PI = ∇.(ν_e ∇ω )
            Grad_Omega_hat_dirx = Kx*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Kx*Omega_hat) )
            Grad_Omega_hat_diry = Ky*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Ky*Omega_hat) )
            PiOmega_hat = Grad_Omega_hat_dirx + Grad_Omega_hat_diry

        elif self.localflag=='from_tau':
            # Calculate the PI term for local: ∇×∇(-2 ν_e S_{ij} )
            Tau11, Tau12, Tau22 = Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky)
            
            Tau11_hat = np.fft.fft2(Tau11)
            Tau12_hat = np.fft.fft2(Tau12)
            Tau22_hat = np.fft.fft2(Tau22)
            
            PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)
        
        # --------- DEBUG MODE ------------------------------------------------
        # #''' test: difference between local  ∇.(ν_e ∇ω ) and not (ν_e ∇.(∇ω)=ν_e ∇^2 ω)
        # c_dynamic_old = coefficient_dleith_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        # Cl_old = c_dynamic_old ** (1/3)
        # eddy_viscosity_old = eddy_viscosity_leith(Cl_old, Delta, characteristic_Omega)
        # Grad_Omega_hat_old = eddy_viscosity_old *(Ksq*Omega_hat)

        # PiOmega_hat_tau = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)

        # import matplotlib.pyplot as plt
        # plt.rcParams['figure.dpi'] = 350
        # VMIN, VMAX = -2, 2
        # fig, axes = plt.subplots(2,3, figsize=(12,8))
        # plt.subplot(2,3,1)
        # plt.title(r'$\Pi=\nu_e \nabla.(\nabla \omega)$, Leith (domain average)')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,2)
        # plt.title(r'$\Pi=\nabla.(\nu_e \nabla \omega)$, Leith (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        
        # plt.subplot(2,3,3) #  ∇×∇(-2 ν_e S_{ij} )
        # plt.title(r'$\Pi=\nabla \times \nabla ( -2 \nu_e \overline{S}_{ij})$, Leith (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat_tau).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,4)
        # plt.title(r'$\nu_e \nabla.(\nabla \omega) - \nabla.(\nu_e \nabla \omega) $')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real-np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        
        # plt.subplot(2,3,6)
        # plt.title(r'$C_L$')
        # plt.pcolor(Cl,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,5)
        # plt.title(r'$Local, \nu_e(x,y)$')
        # plt.pcolor(eddy_viscosity,cmap='gray_r');plt.colorbar()
        # # plt.subplot(2,3,6)
        # # plt.title(r'$\nu_e$')
        # # plt.pcolor(eddy_viscosity, cmap='gray_r');plt.colorbar()
        
        
        # for i, ax in enumerate(axes.flat):
        #     # Set the aspect ratio to equal
        #     ax.set_aspect('equal')

        # plt.show()
        # stop_test

        #PiOmega_hat is instead replaced
        eddy_viscosity = 0
        Cl = 0
        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl

        return PiOmega_hat, eddy_viscosity, Cl

    def PiOmegaGM2_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        PiOmega = PiOmegaGM2_2DFHIT(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        PiOmega_hat = np.fft.fft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity

        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM4_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        PiOmega = PiOmegaGM4_2DFHIT(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        PiOmega_hat = np.fft.fft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity

        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM6_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        PiOmega = PiOmegaGM6_2DFHIT(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        PiOmega_hat = np.fft.fft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def cnn_method(self, model, input_data, Kx, Ky, Ksq):
        """Perform the CNN calculation."""

        # U_hat, V_hat = Psi2UV_2DFHIT(Psi1_hat, Kx, Ky, Ksq)
        # U = np.real(np.fft.ifft2(U_hat))
        # V = np.real(np.fft.ifft2(V_hat))
        # input_data = np.stack((U, V), axis=0)
        output = evaluate_model(model, input_data)

        # Tau11CNN_hat = np.fft.fft2(output[0])
        # Tau12CNN_hat = np.fft.fft2(output[1])
        # Tau22CNN_hat = np.fft.fft2(output[2])

        # PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11CNN_hat, Tau12CNN_hat, Tau22CNN_hat, Kx, Ky, Ksq)

        return output


    def gan_method(self, data):
        # Implement GAN method
        return "Calculated with GAN method"

# Usage example:

# model = PiOmegaModel()  # Default method is NO-SGS
# model.set_method('CNN')
# result = model.calculate("Sample Data")
# print(result)  # Output: Calculated with NO-SGS method

