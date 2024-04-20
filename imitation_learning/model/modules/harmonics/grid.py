from typing import Tuple
import numpy.typing as npt

import numpy as np

def grid1D(boxsize: float, ngrid: int, origin: float=0.) -> Tuple[npt.NDArray, npt.NDArray]:
    """ Returns the x coordinates of a cartesian grid.

    Args:
        boxsize : Box size.
        ngrid : Grid division along one axis.
        origin : Start point of the grid.
    """
    xedges = np.linspace(0., boxsize, ngrid + 1) + origin
    x = 0.5*(xedges[1:] + xedges[:-1])
    return xedges, x

def polargrid(Rmax: float, Nr: int, Nphi: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """ Returns a 2D polar grid.

    Args:
        Rmax - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
    """
    redges, r = grid1D(Rmax, Nr)
    pedges, p = grid1D(2.*np.pi, Nphi)
    p2d, r2d = np.meshgrid(p, r, indexing='ij')
    return p2d, r2d

def wrap_polar(f: npt.NDArray) -> npt.NDArray:
    """ Wraps polar grid, which is useful for plotting purposes.

    Args:
        f - Field polar grid.
    """
    return np.concatenate([f, np.array([f[0]])])

def unwrap_polar(f: npt.NDArray) -> npt.NDArray:
    """ Unwraps polar grid.

    Args:
        f - Wrapped field polar grid.
    """
    return f[:-1]

def wrap_phi(p2d: npt.NDArray) -> npt.NDArray:
    """ Wraps polar grid, which is useful for plotting purposes.

    Args:
        p2d - Phi grid.
    """
    p2d = wrap_polar(p2d)
    p2d[-1] = 2.*np.pi
    return p2d

def unwrap_phi(f: npt.NDArray) -> npt.NDArray:
    """ Unwraps polar grid.

    Args:
        p2d - Wrapped Phi grid.
    """
    return p2d[:-1]
