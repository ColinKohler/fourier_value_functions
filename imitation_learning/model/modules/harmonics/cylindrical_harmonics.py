import math
import torch

from imitation_learning.model.modules.harmonics.harmonics import HarmonicFunction
from imitation_learning.model.modules.harmonics import bessel, grid

def get_n(Nmax: int) -> torch.Tensor:
    """ Returns an array of n up to Nn.

    Args:
       NMax - Maximum n.
    """
    return torch.arange(Nmax+1)[1:]

def get_m(Nm: int) -> torch.Tensor:
    """ Returns m for the 1D angular cylinder components.

    Args:
        Nm - Length of phi grid.
    """
    m = torch.arange(Nm+1)
    return m

def get_k(Nk: int) -> torch.Tensor:
    """ Returns k for the 1D axial cylindrical components.

    Args:
        Nk - Length of the k grid.
    """
    k = torch.arange(Nk+1)[1:]
    return k

def get_knm(xnm: torch.Tensor, Rmax: float) -> torch.Tensor:
    """ Returns the Fouirer Mode k components given the zeros and maximum radius

    Args:
        xnm - Location of zeros.
        Rmax: Maximum radius.
    """
    return xnm / Rmax

def get_Nnm_zero(m: int, xnm: torch.Tensor, Rmax: float) -> torch.Tensor:
    """ Returns the normalization constant for zero-value boundaries.

    Args:
        m - Order.
        xnm - Location of zeros for zero-value boundaries.
        Rmax - Maximum radius.
    """
    Nnm = (Rmax**2. / 2.) * torch.from_numpy(bessel.get_Jm(m+1, xnm.numpy()))**2.
    return Nnm

def get_Nnm_deri(m: int, xnm: torch.Tensor, Rmax: float) -> torch.Tensor:
    """ Returns the normalization constant for derivative boundaries.

    Args:
        m - Order
        xnm - Location of zeros for derivative boundaries.
        Rmax - Maximum radius.
    """
    return (Rmax**2. / 2.) * (1. - m**2. / xnm**2.) * torch.from_numpy(bessel.get_Jm(m, xnm.numpy()))**2.

def get_Rnm(r: torch.Tensor, m: int, knm: float, Nnm: float) -> torch.Tensor:
    """ Radial component of the cylinder basis function.

    Args:
        r - Radial values.
        m - Order.
        knm - Corresponding k Fourier mode for n and m.
        Nnm - Corresponding normalisation constant.
    """
    return (1. / math.sqrt(Nnm)) * torch.from_numpy(bessel.get_Jm(m, knm * r.cpu().numpy()))

def get_Phi_m(m: int, phi: torch.Tensor) -> torch.Tensor:
    """ Angular component of the cylinder basis function.

    Args:
        m - Order.
        phi - Angular values (radians).
    """
    if m == 0:
        return torch.ones_like(phi) / math.sqrt(2 * torch.pi)
    else:
        return torch.stack([
            torch.cos(m * phi) / math.sqrt(2 * torch.pi),
            torch.sin(m * phi) / math.sqrt(2 * torch.pi)
        ])

def get_Z_k(k: int, z: torch.Tensor) -> torch.Tensor:
    """ Axial component of the cylinder basis function.

    Args:
        k - Order.
        z - Axial values.
    """
    return torch.sin(k * torch.pi * z)

def get_Psi_nm(
    n: int,
    m: int,
    k: int,
    r: torch.Tensor,
    phi: torch.Tensor,
    z: torch.Tensor,
    knm: float,
    Nnm: torch.Tensor
) -> torch.Tensor:
    """ Cylinder radial basis function
    Args:
        n - Number of zeros.
        m - Bessel order.
        r - Radius.
        phi - Angle.
        knm - Corresponding k Fourier mode for n and m.
        Nnm - Corresponding normalisation constant.
    """
    Phi_m = get_Phi_m(m, phi)
    Rnm = get_Rnm(r, m, knm, Nnm).to(r.device)
    Z_k = get_Z_k(k, z)
    Psi_nm = Phi_m * Rnm * Z_k

    return Psi_nm

class CylindricalHarmonics(HarmonicFunction):
    def __init__(
        self,
        radial_frequency: int,
        angular_frequency: int,
        axial_frequency: int,
        max_radius: float,
        max_height: float,
        num_radii: int=None,
        num_phi: int=None,
        num_height: int=None,
        boundary: str="zero"
    ):
        super().__init__()

        self.radial_frequency = radial_frequency
        self.angular_frequency = angular_frequency
        self.axial_frequency = axial_frequency
        self.max_radius = max_radius
        self.max_height = max_height
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.num_height = num_height
        self.boundary = boundary

        self.init()

    def init(self) -> None:
        """ Initialize the intermediate variables and basis functions. """
        self.r2d, self.p2d, self.z2d = grid.cylinder_grid(self.max_radius, self.max_height, self.num_radii, self.num_phi, self.num_height)
        self.dr = self.r2d[0][1] - self.r2d[0][0]
        self.dphi = self.p2d[1][0] - self.p2d[0][0]
        self.dh = self.z2d[0][1] - self.z2d[0][0]

        self.m = get_m(self.angular_frequency)
        self.n = get_n(self.radial_frequency)
        self.k = get_k(self.axial_frequency)
        self.m2d, self.n2d, self.k2d = torch.meshgrid(self.m, self.n, self.k, indexing='ij')

        self.xnm = torch.zeros(self.m2d.shape)
        self.knm = torch.zeros(self.m2d.shape)
        self.Nnm = torch.zeros(self.m2d.shape)

        # Compute intermediate variables for Cylinder Basis Functions
        len_m = len(self.m2d)
        for i in range(len_m):
            mval = self.m[i].item()
            nval = self.n[-1].item()
            if self.boundary == "zero":
                xnm = torch.from_numpy(bessel.get_Jm_zeros(mval, nval))
                knm = get_knm(xnm, self.max_radius)
                Nnm = get_Nnm_zero(mval, xnm, self.max_radius)
            else:
                xnm = torch.from_numpy(bessel.get_dJm_zeros(mval, nval))
                knm = get_knm(xnm, self.max_radius)
                Nnm = get_Nnm_deri(mval, xnm, self.max_radius)

            self.xnm[i] = xnm
            self.knm[i] = knm
            self.Nnm[i] = Nnm

        self.m2d_flat = self.m2d.flatten()
        self.n2d_flat = self.n2d.flatten()
        self.k2d_flat = self.k2d.flatten()
        self.xnm_flat = self.xnm.flatten()
        self.knm_flat = self.knm.flatten()
        self.Nnm_flat = self.Nnm.flatten()

        # Pre-Compute Cylinder Basis Functions for specified grid
        self.Psi = torch.zeros(((self.radial_frequency * (self.angular_frequency*2+1)),) * self.axial_frequency + self.r2d.shape)
        li = 0
        for i  in range(0,len(self.m2d_flat)):
            Psi_nm = get_Psi_nm(
                self.n2d_flat[i].item(),
                self.m2d_flat[i].item(),
                self.k2d_flat[i].item(),
                self.r2d,
                self.p2d,
                self.z2d,
                self.knm_flat[i].item(),
                self.Nnm_flat[i]
            )

            if self.m2d_flat[i] == 0:
                self.Psi[li] = Psi_nm
                li+=1
            else:
                self.Psi[li] = Psi_nm[0]
                self.Psi[li+1] = Psi_nm[1]
                li+=2
        self.Psi = self.Psi.unsqueeze(0)

    def evaluate(
        self,
        Pnm: torch.Tensor,
        radii: torch.Tensor=None,
        phis: torch.Tensor=None,
        zs: torch.Tensor=None
    ) -> torch.Tensor:
        """ Evaluate the cylinder fourier coefficients on the predefined grid.

        Args:
            Pnm - Cylinder fourier coefficients.
        """
        B = Pnm.size(0)

        if radii is not None:
            f = torch.zeros((B,) + radii.shape[1:]).to(Pnm.device)
            li = 0
            for i in range(len(self.m2d_flat)):
                Psi = get_Psi_nm(
                    self.n2d_flat[i].item(),
                    self.m2d_flat[i].item(),
                    self.k2d_flat[i].item(),
                    radii,
                    phis,
                    zs,
                    self.knm_flat[i].item(),
                    self.Nnm_flat[i]
                ).to(radii.device)

                if self.m2d_flat[i] == 0:
                    f += torch.einsum("n,nrpz->nrpz", Pnm[:,li], Psi)
                    li += 1
                else:
                    f += torch.einsum("n,nrpz->nrpz", Pnm[:,li], Psi[0])
                    f += torch.einsum("n,nrpz->nrpz", Pnm[:,li+1], Psi[1])
                    li += 2
            return f
        else:
            Psi = self.Psi.repeat(B,1,1,1,1).to(Pnm.device)
            f = torch.zeros((B,) +  self.r2d.shape).to(Pnm.device)
            for i in range(Pnm.size(1)):
                f += torch.einsum("n,nrpz->nrpz", Pnm[:,i], Psi[:,i])
            return f
