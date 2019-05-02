# pylint: disable=E1101, C0103, R0912, R0913, R0914, R0915, W0212

#Copyright 2016 United States Government as represented by the Administrator
#of the National Aeronautics and Space Administration. All Rights Reserved.

'''This module implements the polspice code in Python. This mainly a port
of the PolSpice code by Challinor, Chon, Colombi, Hivon, Prunet, and Szapudi
to Python.'''

import numpy as np
import healpy as H
import scipy.special
from mpi4py import MPI

import cmb_analysis.util.wignercoupling as wc

comm = MPI.COMM_WORLD

def spice(map1, map2=None, window=None, mask=None, window2=None, mask2=None,
          apodizesigma=0.0, apodizetype=0, thetamax=180.0, decouple=False,
          returnxi=False, remove_monopole=False, remove_dipole=False):
    '''This is an implementation of the PolSpice algorithm in Python.

    Parameters
    ----------
    map1 : str or list
        Input filename to load or list of 3 Healpix maps (I, Q, U)

    map2 : str or list, optional
        Input filename to load or list of 3 Healpix maps (I, Q, U)

    window : str or numpy.ndarray, optional
        Filename or array giving the weighted window to use.

    mask : str or numpy.ndarray, optional
        Filename or array giving the mask to use

    window2 : str or numpy.ndarray, optional
        Filename or array giving the weighted window to use for
        the second map. If not input, the same window is used as
        for the first map.

    mask2 : str or numpy.ndarray, optional
        Filename or array giving the mask to use for the second
        map. If not input, the same window is used as for the
        first map.

    apodizesigma : float
        Scale factor of the correlation function apodization (degrees)

    apodizetype : int
        Type of apodization. 0 is a Gaussian window. 1 is a cosine window

    thetamax : float
        maximum value of theta used in the integrals to calculate the
        power spectra from the correlation functions.

    decouple : bool
        whether to compute the decoupled correlation functions

    returnxi : bool
        whether or not to return the correlation functions as an additional
        output of the function

    remove_monopole : bool
        whether or not to remove the monopole from the maps before analyzing
        them

    remove_dipole : bool
        whether or not to remove the dipole from the maps before analyzing
        them

    Returns
    -------
    cls : numpy.ndarray
        An array containing the power spectra of the maps. TT, EE, BB, TE, EB, TB

    xi : numpy.ndarray, optional
        An array containing the real space correlation functions.

    Notes
    -----
    If neither a window nor a mask are input, the functionality will
    be similar to anafast.
    '''

    #If input is a string assume it is a filename
    if isinstance(map1, str):
        try:
            map1 = H.read_map(map1, field=(0, 1, 2))
        except:
            raise ValueError('Input map should have I, Q, and U')

    have_map2 = False
    if isinstance(map2, str):
        try:
            map2 = H.read_map(map2, field=(0, 1, 2))
        except:
            raise ValueError('Input map should have I, Q, and U')
        have_map2 = True

    if isinstance(window, str):
        window = H.read_map(window)
    elif window is None:
        window = np.ones_like(map1[0])

    if mask is None:
        mask = np.ones_like(map1[0])

    #Merge masks/windows
    window = window * mask

    map1 = (map1[0]*window, map1[1]*window, map1[2]*window)

    if have_map2:
        if isinstance(window2, str):
            window2 = H.read_map(window2)
        elif window2 is None:
            window2 = window

        if isinstance(mask2, str):
            mask2 = H.read_map(mask2)
        elif mask2 is None:
            mask2 = mask

        window2 = window2 * mask2

        map2 = (map2[0]*window2, map2[1]*window2, map2[2]*window2)

    #Remove the monopole and dipole. Points outside the mask/window are set
    #to the UNSEEN value so they are ignored when calculating the monopole/
    #dipole
    if remove_monopole:
        idx = window == 0
        map1[0][idx] = H.UNSEEN
        map1[1][idx] = H.UNSEEN
        map1[2][idx] = H.UNSEEN
        map1[0][:] = H.remove_monopole(map1[0], verbose=False)
        map1[1][:] = H.remove_monopole(map1[1], verbose=False)
        map1[2][:] = H.remove_monopole(map1[2], verbose=False)
        map1[0][idx] = 0.0
        map1[1][idx] = 0.0
        map1[2][idx] = 0.0
        if have_map2:
            idx = window2 == 0
            map2[0][idx] = H.UNSEEN
            map2[1][idx] = H.UNSEEN
            map2[2][idx] = H.UNSEEN
            map2[0][:] = H.remove_monopole(map2[0], verbose=False)
            map2[1][:] = H.remove_monopole(map2[1], verbose=False)
            map2[2][:] = H.remove_monopole(map2[2], verbose=False)
            map2[0][idx] = 0.0
            map2[1][idx] = 0.0
            map2[2][idx] = 0.0

    if remove_dipole:
        idx = window == 0
        map1[0][idx] = H.UNSEEN
        map1[1][idx] = H.UNSEEN
        map1[2][idx] = H.UNSEEN
        map1[0][:] = H.remove_dipole(map1[0], verbose=False)
        map1[1][:] = H.remove_dipole(map1[1], verbose=False)
        map1[2][:] = H.remove_dipole(map1[2], verbose=False)
        map1[0][idx] = 0.0
        map1[1][idx] = 0.0
        map1[2][idx] = 0.0
        if have_map2:
            idx = window2 == 0
            map2[0][idx] = H.UNSEEN
            map2[1][idx] = H.UNSEEN
            map2[2][idx] = H.UNSEEN
            map2[0][:] = H.remove_dipole(map2[0], verbose=False)
            map2[1][:] = H.remove_dipole(map2[1], verbose=False)
            map2[2][:] = H.remove_dipole(map2[2], verbose=False)
            map2[0][idx] = 0.0
            map2[1][idx] = 0.0
            map2[2][idx] = 0.0

    #Construction of the cls of the maps and masks/weights
    cls = H.anafast(map1, map2=map2)
    thetas, xi, w, x = get_xi_from_cl(cls, return_leggauss=True, thetamax=thetamax)

    if window is None:
        wls = None
    else:
        wls = H.anafast(window, map2=window2)
        #Compute xi from the cls of the map and masks
        thetas, xi_mask = get_xi_from_cl(wls, thetas=thetas)

        xi_final = _correct_xi_from_mask(xi, xi_mask)

    if decouple:
        xi_final = _update_xi(xi_final, x, thetamax, cls, cl_mask=wls)

    ncor = len(xi_final)

    #apodize the correlation function
    if apodizesigma > 0.0:
        apfunction = _apodizefunction(x, apodizesigma, thetamax, apodizetype)
        for i in range(ncor):
            xi_final[i, :] *= apfunction

    #compute the cls from xi
    cls_out = do_cl_from_xi(xi_final, w, x, decouple=decouple,
                            apodizesigma=apodizesigma,
                            apodizetype=apodizetype,
                            thetamax=thetamax)

    #Correct cl for beam, pixel, transfer function???

    if returnxi:
        return cls_out, xi_final
    else:
        return cls_out

def _correct_xi_from_mask(xi, xi_mask):
    '''Correct the amplitude of the correlation function due to the
    presence of the mask/window

    Parameters
    ----------
    xi : array
        The correlation function of the input maps

    xi_mask : array
        The correlation function of the mask/window

    Returns
    -------
    xi_output : array
        The correlation function of the input maps
        corrected for the effect of the presence of
        the mask.
    '''

    if len(xi_mask) > 20:
        ncmask = 1
        xi_mask.shape = (1, len(xi_mask))
    else:
        ncmask = len(xi_mask)

    if len(xi) > 20:
        ncor = 1
        xi.shape = (1, len(xi))
    else:
        ncor = len(xi)

    if ncmask == 3:
        kindex = np.array([0, 1, 1, 2, 2, 1, 2, 2, 1], dtype=np.int)
    elif ncmask == 4:
        kindex = np.array([0, 1, 1, 2, 2, 1, 3, 3, 1], dtype=np.int)
    else:
        kindex = np.zeros(9, dtype=np.int)

    ncountbinrm = np.zeros(9)
    xi_final = np.empty_like(xi)

    for j in range(ncor):
        kmask = kindex[j]
        idx = xi_mask[kmask, :] != 0
        xi_final[j, idx] = xi[j, idx] / xi_mask[kmask, idx]
        #xi_final[j, xi_mask[kmask, :] == 0] = 0
        xi_final[j, ~idx] = 0
        ncountbinrm[j] += np.sum(~idx)

    if ncmask == 1:
        xi_mask.shape = (np.size(xi_mask), )

    if ncor == 1:
        xi.shape = (np.size(xi), )

    return xi_final

def _apodizefunction(mu, fwhm_degrees, thetamax, typef=0):
    '''Calculates an apodization function to apply to
    the correlation functions

    Parameters
    ----------
    mu: array
        cos(thetas), where thetas is the angles at which the
        correlation function is calculated

    fwhm_degrees: float
        The full width half max in degrees for the apodization

    thetamax: float
        The maximum value of theta in the correlation function

    typef: int
        Either 0 or 1. Decides between two different types of
        apodization

    Returns
    ------
    output: array
        The apodized function
    '''

    if fwhm_degrees <= 0:
        return np.ones_like(mu)

    output = np.zeros_like(mu)
    if typef == 0:
        apodizesigma = np.radians(fwhm_degrees)/np.sqrt(8*np.log(2))

        idx = np.degrees(np.arccos(mu)) < thetamax
        tmp = np.exp(-(np.arccos(mu)/apodizesigma)**2/2.0)
        output[idx] = tmp[idx]
    elif typef == 1:
        apodizesigma = np.radians(fwhm_degrees)
        val1 = np.degrees(np.arccos(mu))/thetamax
        val2 = np.arccos(mu)/apodizesigma
        argument = np.max([val1, val2], axis=0)

        idx = argument < 1.0

        tmp = (1.0 + np.cos(argument*np.pi)) / 2.0
        output[idx] = tmp[idx]

    return output

def _get_legendre(x, ell, s=None, Plm12=None):
    '''Calculate the functions of Legendre polyonmials that are needed
    for the correlation function transformations and used in several
    other functions.

    Parameters
    ----------
    x : array
        An array of values of cos(theta) over which to calculate the
        Legengre polynomial functions

    ell : int
        The ell value for the Legendre polynomials

    s : array, optional
        The values of sin(theta)

    Plm12 : array, optional
        The values of P_{ell-1}^2 at each value of x. If not input,
        these will be calculated.

    Returns
    -------
    Pl0 : array
        Legendre polynomial of order ell evaluated at x

    Pl2 : array
        Associated legendre polynomial (m=2) of order ell evaluated at x

    Gp : array
        Needed to calculate the small d-matrix used for the polarization
        correlation function calculations

    Gm : array
        Needed to calculate the small d-matrix used for the polarization
        correlation function calculations

    Nl : array
        sqrt((ell-2)!/(ell+2)!)
    '''

    m = 2
    if s is None:
        s = np.sqrt(1 - x**2)

    dl = 1.0*ell
    Pl0 = scipy.special.lpmv(0, ell, x)
    Pl2 = scipy.special.lpmv(2, ell, x)

    #It is faster to input P_(l-1)^2 than to calculate it when
    #x is a large array and there is a good chance this function
    #will be repeatedly called with increasing ell
    if Plm12 is None:
        Plm12 = scipy.special.lpmv(2, ell-1, x)

    Gp = np.zeros_like(Pl0)
    Gm = np.zeros_like(Pl0)
    Nl = 1.0

    if ell >= 2:
        idx = (x == 1.0)
        Gp[idx] = 0.25
        Gm[idx] = 0.25

        idx = (x == -1.0)
        Gp[idx] = Pl0[idx]*0.25
        Gm[idx] = -Pl0[idx]*0.25

        idx = (np.abs(x) != 1.0)
        x = x[idx]
        s = s[idx]

        #sqrt((l-2)!/(l+2)!)
        Nl = 1.0 / np.sqrt((dl+2.0)*(dl+1.0)*dl*(dl-1.0))

        # (l+2)* cot(theta)/sin(theta) * P_{\ell-1}^2
        t1 = (dl+m) * x/(s*s) * Plm12[idx]

        # ((l-4)/sin(theta)^2 + (l-1)*l*0.5) * P_\ell^2
        t2 = ((dl-m*m)/(s*s) + (dl-1.0)*dl*0.5) * Pl2[idx]

        # (l-1) * cos(theta) * P_\ell^2
        t3 = (dl-1.0) * x * Pl2[idx]

        # (l+2) * P_{\ell-1}^2
        t4 = (dl+m) * Plm12[idx]

        Gp[idx] = Nl**2 * (t1-t2)
        Gm[idx] = Nl**2 * (t3-t4) * m/(s*s)

    return Pl0, Pl2, Gp, Gm, Nl

def do_cl_from_xi(xi, w, x, decouple=False, apodizesigma=0.0,
                  thetamax=180.0, apodizetype=0):
    '''Calculate the C_l from the real-space correlation functions

    Parameters
    ----------
    xi : array
        The correlation function

    w : array
        The weights used in the Gauss-Legendre quadrature

    x : array
        The x values used in the Gauss-Legendre quadrature

    decouple : bool
        Whether or not the input correlation function is the decoupled
        polarization correlation function

    Note
    ----
    This does the integrals by Gauss-Legendre quatrature, therefore the
    real-space correlation functions must be input at the correct x values
    along with the correct weights to use.
    '''

    if len(xi) > 20:
        nell = len(xi)
        ncl = 1
    else:
        #ncl = len(xi)
        ncl = 6
        nell = len(xi[0])

    cls = np.empty([ncl, nell])
    Fl = np.empty(nell)

    nfact = 2

    Pl2 = np.zeros_like(x)

    #Calculation of the non ell dependent part of the integrand of
    #Equation 65
    apfunction = _apodizefunction(x, apodizesigma, thetamax, apodizetype)
    tempo = apfunction / np.sin(np.arccos(x)/2.0)**2 * 2*w

    for i in range(nell):
        Pl0, Pl2, Gp, Gm, Nl = _get_legendre(x, i, Plm12=Pl2)

        cls[0, i] = np.sum(w*xi[0, :]*Pl0)*2*np.pi

        if ncl > 1:
            wp = nfact**2 * w * xi[1, :] * np.pi
            wm = nfact**2 * w * xi[2, :] * np.pi
            wx = nfact**2 * w * xi[5, :] * np.pi

            if decouple:
                dl2m2 = Gp-Gm
                cls[1, i] = np.sum(wp*dl2m2)
                cls[2, i] = np.sum(wm*dl2m2)
                Fl = np.sum(tempo*dl2m2)
                if Fl > 0:
                    cls[1, i] /= Fl
                    cls[2, i] /= Fl
            else:
                cls[1, i] = np.sum(wp*Gp + wm*Gm)
                cls[2, i] = np.sum(wm*Gp + wp*Gm)

            #Need an apodization correction like Fl for EE and BB
            cls[3, i] = np.sum(Nl*Pl2*nfact*w*xi[3, :] * np.pi)
            cls[4, i] = np.sum(Nl*Pl2*nfact*w*xi[4, :] * np.pi)
            cls[5, i] = np.sum(wx*(Gp-Gm))

    cross_correction = False
    if cross_correction:
        fact = _correct_TE(apfunction, x, w)
        cls[3, :] *= fact
        cls[4, :] *= fact

    if ncl == 1:
        cls.shape = (nell, )

    return cls

def get_xi_from_cl(cls, thetas=None, return_leggauss=False, thetamax=180.0):
    '''Calculate of the real space correlation function from the C_ls.

    Parameters
    ----------
    cls : array
        Input Cl power spectra

    thetas : array, optional
        Angles at which to calculate the correlation function. If not input,
        they will be calculated at the points needed for Legendre-Gauss
        quadrature

    return_leggauss : bool
        Whether to additionally return the x values and weights for
        Legendre-Gauss quadrature. If set to True, the input thetas is
        ignored

    thetamax : float
        Maximum value of theta to calculate for the correlation function.
        Modified the Legendre-Gauss quadrature x values and weights.
        Ignored if thetas is input and return_leggauss is False.

    Returns
    -------
    thetas_out : array
        The angles at which the correlation function is calculated. Same
        as the input thetas if it is input and return_leggauss is False.

    xi : array
        The calculated correlation function

    w : array, optional
        The weights needed for Legendre-Gauss quadrature. Output if
        return_leggauss is True

    x : array, optional
        cos(thetas). Output if return_leggauss is True
    '''

    #Calculation of the correlation function
    if len(cls) > 20:
        ncl = 1
        nell = len(cls)
    else:
        ncl = len(cls)
        nell = len(cls[0])

    ell_vect = np.arange(nell)

    if return_leggauss or thetas is None:
        ntheta = nell
        x, w = np.polynomial.legendre.leggauss(ntheta)
        xmin = np.cos(np.radians(thetamax))
        xmax = 1.0
        x = (xmax-xmin)/2.0 * x + (xmax+xmin)/2.0
        w = (xmax-xmin)/2.0 * w

        thetas = np.arccos(x)
    else:
        ntheta = len(thetas)

    x = np.cos(thetas)

    Pl2 = np.zeros_like(x)

    xi = np.zeros([ncl, ntheta])

    for ell in ell_vect:
        Pl0, Pl2, Gp, Gm, Nl = _get_legendre(x, ell, Plm12=Pl2)

        twolp1 = 2*ell + 1
        i = ell
        if ncl > 1:
            xi[0, :] += cls[0][i] * Pl0 * twolp1
        else:
            xi[0, :] += cls[i] * Pl0 * twolp1

        if ell >= 2 and ncl > 1:
            xi[1, :] += 2.0*(cls[1][i]*Gp + cls[2][i]*Gm)*twolp1
            xi[2, :] += 2.0*(cls[2][i]*Gp + cls[1][i]*Gm)*twolp1
            xi[3, :] += cls[3][i] * Nl*Pl2 * twolp1
            xi[4, :] += cls[4][i] * Nl*Pl2 * twolp1
            xi[5, :] += 2.0*cls[5][i]*(Gp - Gm)*twolp1

    xi /= 4*np.pi

    if ncl == 1:
        xi.shape = (ntheta,)

    if return_leggauss:
        return thetas, xi, w, x
    else:
        return thetas, xi

def _correct_TE(apfunction, x, w):
    '''Calculate an amplitude correction of the TE cross-correlation
    due to the apodization of the correlation function.

    Parameters
    ----------
    apfunction : array
        The function dependent on angle used to apodize the correlation
        function.

    x : array
        cos(angle), the Legendre-Gauss quadrature points

    w : array
        the weights needed for the Legendre-Gauss quadrature

    Returns
    -------
    fact : array
        TE correction function due to the fact that the correlation function
        is apodized
    '''

    nell = len(apfunction)
    lmax = nell - 1

    #Calculate f_l from apodizing function in real space
    fl = np.zeros(nell)
    for i in range(nell):
        Pl0 = scipy.special.lpmv(0, i, x)
        fl[i] = np.sum(w*apfunction*Pl0)*2*np.pi * (2*i+1) / (4*np.pi)

    kcross = np.zeros([nell, nell])

    rank = comm.rank
    size = comm.size

    for l1 in range(2+rank, nell, size):
        for l2 in range(l1, nell):
            l3min = np.abs(l1-l2)
            l3max = np.abs(l1+l2)

            #runs from l3min to l3max
            wigner00 = wc.wigner3j_vect(2*l1, 2*l2, 2*0, 2*0)
            wigner22 = wc.wigner3j_vect(2*l1, 2*l2, 2*2, -2*2)

            if l3max > lmax:
                tmp = lmax-l3max
                wigner00 = wigner00[:tmp]
                wigner22 = wigner22[:tmp]

            kcross[l1, l2] = np.sum(wigner00*wigner22*fl[l3min:l3max+1])
            kcross[l2, l1] = kcross[l1, l2]

            kcross[l1, l2] *= 2*l2 + 1

            if l2 != l1:
                kcross[l2, l1] *= 2*l1 + 1

    kcross = comm.allreduce(kcross)

    fact = np.sum(kcross, axis=1)

    return fact

def _update_xi(xi, mu, thetamax_degrees, cl, cl_mask=None):
    '''This replaces the polarized elements of xi with versions that only
    depend on E or B polarization, but not both.

    Note
    ----
    This corresponds to Section 5 in Chon, Challinor, Prunet, Hivon, and
    Szapudi (2004)
    '''

    thetamax = np.radians(thetamax_degrees)

    #Figure out how many points we need to do accurate Simpson integration
    #nmax is hardcoded right now since the original PolSpice software used
    #Numerical Recipes code that we cannot distribute
    nmax = 10

    nell = len(mu)
    theta = np.arccos(mu)

    c_beta = np.zeros(nell)

    #Do the integrals in Eq 90 and then construct xi(beta)
    sum1, sum2 = _cumul_simpson(theta, cl, nmax, thetamax, cl_mask=cl_mask)

    c_plus = _cplus(mu, cl, cl_mask=cl_mask)
    c_beta = np.empty_like(c_plus)
    c_beta[:] = c_plus[:]
    c_beta += sum1/np.sin(theta/2.0)**2 - sum2*2.0*(2.0 + np.cos(theta))/np.sin(theta/2.0)**4
    x1 = np.array(xi[1, :])
    x2 = np.array(xi[2, :])

    xi[1, :] = 0.5 * (c_beta+x1-x2)
    xi[2, :] = 0.5 * (c_beta-x1+x2)

    return xi

def _cplus(x, cl, cl_mask=None):
    '''Calculate xi_+ corelation function.

    Note
    ----
    This is Eqs 17 and 9.'''

    if len(np.shape(cl)) != 2:
        raise ValueError("Input Cl must be 2d array (ncl, nell)")

    if len(cl) < 4:
        raise ValueError("Input Cl must have polarization")

    Pl2 = 0.0

    nell = len(cl[0])

    ell_vect = np.arange(nell)
    c_plus = np.zeros_like(x)
    cmask = np.zeros_like(x)

    for ell in ell_vect:
        Pl0, Pl2, Gp, Gm, Nl = _get_legendre(x, ell, Plm12=Pl2)

        dl22 = Gp + Gm

        i = ell
        twolp1 = 2.0*ell + 1.0
        if ell >= 2:
            c_plus += 2.0*(cl[1][i] + cl[2][i])*dl22*twolp1

        if cl_mask is not None:
            cmask += cl_mask[i]*Pl0*twolp1

    if cl_mask is not None:
        idx = cmask > 0.0
        c_plus[idx] /= cmask[idx]
        c_plus[~idx] = 0.0
    else:
        c_plus /= 4*np.pi

    return c_plus

def _cumul_simpson(theta, cl, nmax, thetamax, cl_mask=None):
    '''
    Notes
    -----
    This calculates the two integrals in Eq. 90
    '''

    npoints = 2.0**(nmax-1)
    step = thetamax / npoints

    beta = np.linspace(0.0, thetamax, num=npoints+1)

    c_plus = _cplus(np.cos(beta), cl, cl_mask=cl_mask)
    ftemp1 = np.sin(beta)/np.cos(beta/2.0)**4 * c_plus
    ftemp2 = np.tan(beta/2.0)**3 * c_plus

    nell = len(cl[0])

    sum1 = np.zeros(nell)
    sum2 = np.zeros(nell)
    lastn = 2*np.array(np.round(theta/(2.0*step)), dtype=np.int)

    for j in range(nell):
        if lastn[j] != 0:
            sum1[j] = scipy.integrate.simps(ftemp1[:lastn[j]+1], dx=step)
            sum2[j] = scipy.integrate.simps(ftemp2[:lastn[j]+1], dx=step)

    #Since beta[lastn[j]] is not exactly theta[j], we apply the basic Simpson's rule on
    #the small interval between those two values
    step2 = (theta - step*lastn) / 2.0
    theta0 = theta-step2
    theta1 = theta

    c_plus0 = _cplus(np.cos(theta0), cl, cl_mask=cl_mask)
    c_plus1 = _cplus(np.cos(theta1), cl, cl_mask=cl_mask)

    int1_0 = np.sin(theta0)/np.cos(theta0/2.0)**4 * c_plus0
    int1_1 = np.sin(theta1)/np.cos(theta1/2.0)**4 * c_plus1
    int2_0 = np.tan(theta0/2.0)**3 * c_plus0
    int2_1 = np.tan(theta1/2.0)**3 * c_plus1

    endpoint = step2*(ftemp1[lastn] + 4.0 * int1_0 + int1_1) / 3.0
    sum1 += endpoint

    endpoint = step2*(ftemp2[lastn] + 4.0 * int2_0 + int2_1) / 3.0
    sum2 += endpoint

    return sum1, sum2
