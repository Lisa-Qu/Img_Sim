"""
zernike.py — Pure-numpy Zernike moments (drop-in replacement for mahotas).

Produces identical output to mahotas.features.zernike_moments so that
existing pickle indexes remain compatible.
"""

import numpy as np
from math import factorial, pi


def zernike_moments(im, radius, degree=8, cm=None):
    """Compute Zernike moments of an image.

    Parameters
    ----------
    im : 2-D ndarray
        Input image (grayscale).
    radius : int
        Radius of the unit disk in pixels.
    degree : int
        Maximum polynomial degree (default 8).
    cm : tuple (row, col) or None
        Center of mass.  Computed from the image if *None*.

    Returns
    -------
    ndarray, shape (N,)
        Magnitudes |Z_nl| for every valid (n, l) pair where n-l is even,
        ordered by ascending n then ascending l.
    """
    if cm is None:
        total = im.sum()
        if total == 0:
            c0 = im.shape[0] / 2.0
            c1 = im.shape[1] / 2.0
        else:
            Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
            c0 = (im * Y).sum() / total
            c1 = (im * X).sum() / total
    else:
        c0, c1 = cm

    Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
    P = im.ravel()

    # Normalise coordinates to unit disk
    Yn = (Y.astype(np.float64) - c0) / radius
    Xn = (X.astype(np.float64) - c1) / radius
    Yn = Yn.ravel()
    Xn = Xn.ravel()

    Dn = np.sqrt(Xn ** 2 + Yn ** 2)
    np.maximum(Dn, 1e-9, out=Dn)

    # Mask: inside unit disk AND positive pixel value
    k = (Dn <= 1.0) & (P > 0)

    frac_center = np.array(P[k], dtype=np.float64)
    frac_center /= frac_center.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    Dn = Dn[k]

    # Angle as complex unit vector  e^{i*theta}
    An = (Xn / Dn) + 1j * (Yn / Dn)

    # Precompute An**l for l = 0 .. degree+1
    Ans = [np.ones_like(An)]           # l=0
    Ans.append(An.copy())              # l=1
    for p in range(2, degree + 2):
        Ans.append(An ** p)

    # Collect moments
    zvalues = []
    for n in range(degree + 1):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                z = _znl(Dn, Ans[l], frac_center, n, l)
                zvalues.append(abs(z))

    return np.array(zvalues)


def _znl(D, A_l, frac_center, n, l):
    """Compute a single Zernike moment Z_{n,l}.

    Parameters
    ----------
    D : 1-D ndarray (float64)  — normalised distances
    A_l : 1-D ndarray (complex128) — e^{i*l*theta} per pixel
    frac_center : 1-D ndarray (float64) — normalised pixel weights
    n : int — radial degree
    l : int — angular frequency

    Returns
    -------
    complex — the (unnormalised-magnitude) moment
    """
    # Radial polynomial  R_n^l(d) = sum_m g_m * d^(n-2m)
    V = np.zeros(len(D), dtype=np.complex128)
    for m in range((n - l) // 2 + 1):
        coeff = ((-1) ** m * factorial(n - m)
                 / (factorial(m)
                    * factorial((n - 2 * m + l) // 2)
                    * factorial((n - 2 * m - l) // 2)))
        V += coeff * (D ** (n - 2 * m))

    V *= A_l  # multiply by angular part

    return (n + 1) / pi * np.sum(frac_center * np.conj(V))
