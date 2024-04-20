import numpy.typing as npt

import numpy as np
from scipy import optimize
from scipy.special import jv as scipy_jv
from scipy.special import jvp as scipy_jvp
from scipy.special import jn_zeros as scipy_jn_zeros
from scipy.special import jnp_zeros as scipy_jnp_zeros

def get_Jm(m: int, x: float | npt.NDArray) -> float | npt.NDArray:
    """ Returns the value of the Bessel fns of the 1st kind of order m at x.

    Args:
        m - Bessel fn order.
        x - X-coordinates.
    """
    return scipy_jv(m, x)

def get_dJm(m: int, x: float | npt.NDArray) -> float | npt.NDArray:
    """ Returns the value of the derivative of the Bessel fns of the 1st kind of order m at x.

    Args:
        m - Bessel fn order.
        x - X-coordinates.
    """
    return scipy_jvp(m,x)

def get_Jm_zeros(m:int, n: int) -> npt.NDArray:
    """ Returns the first n zeros of the Bessel fns of the 1st kind of order m.
    Args:
        m - Bessel fn order.
        n - Number of zeros.
    """
    return scipy_jn_zeros(m,n)

def get_dJm_zeros(m:int, n: int) -> npt.NDArray:
    """ Returns the first n zeros of the derrivative of the Bessel fns of the 1st kind of order m.
    Args:
        m - Bessel fn order.
        n - Number of zeros.
    """
    return scipy_jnp_zeros(m,n)
