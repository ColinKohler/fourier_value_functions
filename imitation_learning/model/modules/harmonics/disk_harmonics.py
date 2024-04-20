import numpy.typing as npt

import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

import torch
from torch import nn
from imitation_learning.model.modules.harmonics.harmonics import HarmonicFunction
from imitation_learning.model.modules.harmonics import bessel, grid

def get_n(Nmax: int) -> npt.NDArray:
    """ Returns an array of n up to Nn.

    Args:
       NMax - Maximum n.
    """
    return np.arange(Nmax+1)[1:]

def get_m(Nm: int) -> npt.NDArray:
    """ Returns m for the 1D angular polar components.

    Args:
        Nm - Length of phi grid.
    """
    m = np.arange(Nm+1)
    #condition = np.where(m > float(Nm+1) / 2.)[0]
    #m[condition] -= Nm
    return m

def get_knm(xnm: npt.NDArray, Rmax: float) -> npt.NDArray:
    """ Returns the Fouirer Mode k components given the zeros and maximum radius

    Args:
        xnm - Location of zeros.
        Rmax: Maximum radius.
    """
    return xnm / Rmax

def get_Nnm_zero(m: int, xnm: npt.NDArray, Rmax: float) -> npt.NDArray:
    """ Returns the normalization constant for zero-value boundaries.

    Args:
        m - Order.
        xnm - Location of zeros for zero-value boundaries.
        Rmax - Maximum radius.
    """
    Nnm = ((Rmax**2.)/2.)*(bessel.get_Jm(m+1, xnm)**2.)
    return Nnm

def get_Nnm_deri(m: int, xnm: npt.NDArray, Rmax: float) -> npt.NDArray:
    """ Returns the normalization constant for derivative boundaries.

    Args:
        m - Order
        xnm - Location of zeros for derivative boundaries.
        Rmax - Maximum radius.
    """
    return ((Rmax**2.)/2.)*(1. - (m**2.)/(xnm**2.))*(bessel.get_Jm(m, xnm)**2.)

def get_Rnm(r: npt.NDArray, m: int, knm: float, Nnm: float) -> npt.NDArray:
    """ Radial component of the polar basis function.

    Args:
        r - Radial values.
        m - Order.
        knm - Corresponding k Fourier mode for n and m.
        Nnm - Corresponding normalisation constant.
    """
    return (1./np.sqrt(Nnm))*bessel.get_Jm(m, knm*r)

def get_Phi_m(m: int, phi: npt.NDArray) -> npt.NDArray:
    """ Angular component of the polar basis function.

    Args:
        m - Order.
        phi - Angular values (radians).
    """
    if m == 0:
        return np.ones_like(phi) / np.sqrt(2*np.pi)
    else:
        return np.stack([np.cos(m * phi) / np.sqrt(2*np.pi), np.sin(m * phi)/ np.sqrt(2*np.pi)])

def get_Psi_nm(n: int, m: int, r: npt.NDArray, phi: npt.NDArray, knm: float, Nnm: npt.NDArray) -> npt.NDArray:
    """ Polar radial basis function
    Args:
        n - Number of zeros.
        m - Bessel order.
        r - Radius.
        phi - Angle.
        knm - Corresponding k Fourier mode for n and m.
        Nnm - Corresponding normalisation constant.
    """
    Phi_m = get_Phi_m(m, phi)
    Rnm = get_Rnm(r, m, knm, Nnm)
    Psi_nm = Phi_m * Rnm

    return Psi_nm

class DiskHarmonics(HarmonicFunction):
    def __init__(
        self,
        radial_frequency: int,
        angular_frequency: int,
        max_radius: float,
        num_radii: int=None,
        num_phi: int=None,
        boundary: str="zero"
    ):
        super().__init__()

        self.radial_frequency = radial_frequency
        self.angular_frequency = angular_frequency
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.boundary = boundary

        self.init()

    def init(self) -> torch.Tensor:
        """ Initialize the intermediate variables and basis functions. """
        self.p2d, self.r2d = grid.polargrid(self.max_radius, self.num_radii, self.num_phi)
        self.dr = self.r2d[0][1] - self.r2d[0][0]
        self.dphi = self.p2d[1][0] - self.p2d[0][0]
        self.m = get_m(self.angular_frequency)
        self.n = get_n(self.radial_frequency)
        self.m2d, self.n2d = np.meshgrid(self.m, self.n, indexing='ij')

        self.xnm = np.zeros(np.shape(self.m2d))
        self.knm = np.zeros(np.shape(self.m2d))
        self.Nnm = np.zeros(np.shape(self.m2d))

        # Compute intermediate variables for Polar Basis Functions
        len_m = len(self.m2d)
        for i in range(len_m):
            mval = self.m[i]
            nval = self.n[-1]
            if self.boundary == "zero":
                xnm = bessel.get_Jm_zeros(mval, nval)
                knm = get_knm(xnm, self.max_radius)
                Nnm = get_Nnm_zero(mval, xnm, self.max_radius)
            else:
                xnm = bessel.get_dJm_zeros(mval, nval)
                knm = get_knm(xnm, self.max_radius)
                Nnm = get_Nnm_deri(mval, xnm, self.max_radius)

            self.xnm[i] = xnm
            self.knm[i] = knm
            self.Nnm[i] = Nnm

        self.m2d_flat = np.copy(self.m2d).flatten()
        self.n2d_flat = np.copy(self.n2d).flatten()
        self.xnm_flat = np.copy(self.xnm).flatten()
        self.knm_flat = np.copy(self.knm).flatten()
        self.Nnm_flat = np.copy(self.Nnm).flatten()

        # Pre-Compute Polar Basis Functions for specified grid
        self.Psi = np.zeros(((self.radial_frequency * (self.angular_frequency*2+1)),) + self.r2d.shape)
        li = 0
        for i  in range(0,len(self.m2d_flat)):
            Psi_nm = get_Psi_nm(self.n2d_flat[i], self.m2d_flat[i], self.r2d, self.p2d, self.knm_flat[i], self.Nnm.flat[i])
            if self.m2d_flat[i] == 0:
                self.Psi[li] = Psi_nm
                li+=1
            else:
                self.Psi[li] = Psi_nm[0]
                self.Psi[li+1] = Psi_nm[1]
                li+=2
        self.Psi = torch.from_numpy(self.Psi).unsqueeze(0)

    def evaluate(
        self,
        Pnm: torch.Tensor,
        radii: torch.Tensor=None,
        phis: torch.Tensor=None
    ) -> torch.Tensor:
        """ Evaluate the polar fourier coefficients on the predefined grid.

        Args:
            Pnm - Polar fourier coefficients.
        """
        if radii is not None:
            basis_fns = self.generate_basis_fns(radii, phis)
            return torch.einsum("inm,inm->i", coeffs, basis_fns.permute(2,0,1))
        else:
            B = Pnm.size(0)
            Psi = self.Psi.repeat(B,1,1,1)
            f = torch.zeros((B,) +  self.r2d.shape)
            for i in range(Pnm.size(1)):
                f += torch.einsum("n,nxy->nxy", Pnm[:,i], Psi[:,i])
            return f
