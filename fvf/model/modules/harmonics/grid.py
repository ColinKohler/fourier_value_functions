from typing import Tuple
import torch

def grid1D(boxsize: float, ngrid: int, origin: float=0.) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns the x coordinates of a cartesian grid.

    Args:
        boxsize : Box size.
        ngrid : Grid division along one axis.
        origin : Start point of the grid.
    """
    xedges = torch.linspace(0., boxsize, ngrid + 1) + origin
    x = 0.5 * (xedges[1:] + xedges[:-1])
    return xedges, x

def polargrid2(
    Rmax: float,
    Nr: int,
    Nphi: int,
    r_origin: float=0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns a 2D polar grid.

    Args:
        Rmax - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
        r_origin - Radius origin point
    """
    T = [Nphi * i for i in range(1,num_radii+1)]
    r_edges, r = grid.grid1D(max_radius, Nr, origin=r_origin)

    rt_pairs = []
    for i in range(len(r)):
        ps = (np.arange(n[i]) * (2*np.pi / n[i])).reshape(n[i], 1)
        rs = np.array([r[i]] * n[i]).reshape(n[i],1)
        rt_pairs.extend(np.concatenate((ps,rs), axis=1).tolist())
    return np.array(rt_pairs)

def polargrid(
    Rmax: float,
    Nr: int,
    Nphi: int,
    r_origin: float=0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns a 2D polar grid.

    Args:
        Rmax - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
        r_origin - Radius origin point
    """
    redges, r = grid1D(Rmax, Nr, origin=r_origin)
    pedges, p = grid1D(2.0 * torch.pi, Nphi)
    r2d, p2d = torch.meshgrid(r, p, indexing='ij')
    return r2d, p2d

def cylinder_grid(
    Rmax: float,
    Zmax: float,
    Nr: int,
    Nphi: int,
    Nz: int,
    r_origin: float=0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns a 3D cylinder grid.

    Args:
        Rmax - Maximum radius.
        Zmax - Maximum height.
        Nr - Number of elements along the radial axis.
        Nphi - Number of elements along the angular axis.
        Nz - Number of elements alongthe axial axis
    """
    redges, r = grid1D(Rmax, Nr, origin=r_origin)
    pedges, p = grid1D(2.0 * torch.pi, Nphi)
    zedges, z = grid1D(Zmax, Nz, origin=r_origin)
    r2d, p2d, z2d = torch.meshgrid(r, p, z, indexing='ij')
    return r2d, p2d, z2d


def wrap_polar(f: torch.Tensor) -> torch.Tensor:
    """ Wraps polar grid, which is useful for plotting purposes.

    Args:
        f - Field polar grid.
    """
    return torch.concatenate([f, torch.array([f[0]])])

def unwrap_polar(f: torch.Tensor) -> torch.Tensor:
    """ Unwraps polar grid.

    Args:
        f - Wrapped field polar grid.
    """
    return f[:-1]

def wrap_phi(p2d: torch.Tensor) -> torch.Tensor:
    """ Wraps polar grid, which is useful for plotting purposes.

    Args:
        p2d - Phi grid.
    """
    p2d = wrap_polar(p2d)
    p2d[-1] = 2.0 * torch.pi
    return p2d

def unwrap_phi(f: torch.Tensor) -> torch.Tensor:
    """ Unwraps polar grid.

    Args:
        p2d - Wrapped Phi grid.
    """
    return p2d[:-1]
