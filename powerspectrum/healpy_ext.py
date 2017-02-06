# pylint: disable=E1101, C0103, R0912, R0913, R0914, R0915, W0212

#Copyright 2016 United States Government as represented by the Administrator
#of the National Aeronautics and Space Administration. All Rights Reserved.


'''This module provides extensions to certain Healpy routines. Some Healpy
routines are now allowed to take 4 Stokes parameters. In addition, code to
calculate the pure Cls are provided.

Most of the functions to calculate the pure Cl is ported and modified from
the pure s2hat code.
'''

import numpy as np
import healpy as H


def window2vecttens(window_scal, mask=None, lmax=None, mmax=None):
    '''
    Calculates the vector/tensor windows needed in the pure Cl
    calculation

    Notes
    -----
    We calculate both straight from the wlms of the scalar window because
    calculating the spin-2 from the spherical harmonic transform of a
    spin-1 window will result in larger errors.

    If you don't want zero pixels to be masked out input a full sky mask.
    '''

    nside = H.npix2nside(len(window_scal))

    if mask is None:
        mask = np.ones_like(window_scal)
        #mask[window_scal == 0] = 0
    else:
        window_scal = window_scal * mask

    wlm = H.map2alm(window_scal, lmax=lmax, mmax=mmax)

    if lmax is None:
        lmax = H.Alm.getlmax(len(wlm))

    spin = 1
    wlm_tmp = wlm_scalar2spin(wlm, spin)
    window_vect = H.alm2map_spin(wlm_tmp, nside, spin, lmax, mmax=mmax)
    window_vect[0] *= mask
    window_vect[1] *= mask

    spin = 2
    wlm_tmp = wlm_scalar2spin(wlm, spin)
    window_tens = H.alm2map_spin(wlm_tmp, nside, spin, lmax, mmax=mmax)
    window_tens[0] *= mask
    window_tens[1] *= mask

    return window_vect, window_tens


def wlm_scalar2spin(wlm, spin):
    '''

    Notes
    -----
    Input wlm should be wlm of the scalar map and not the s=0 wlm which
    differ by a negative sign.
    '''

    if isinstance(wlm, tuple) or isinstance(wlm, list):
        lmax = H.Alm.getlmax(wlm[0].size)
    else:
        lmax = H.Alm.getlmax(wlm.size)

    ell, m_tmp = H.Alm.getlm(lmax)

    fact = -_nfunc(ell, spin)

    if isinstance(wlm, tuple) or isinstance(wlm, list):
        wlm = (wlm[0]*fact, wlm[1]*fact)
    else:
        wlm_0 = np.zeros_like(wlm)
        wlm = (wlm*fact, wlm_0)

    return wlm


def _nfunc(ells, spin):
    '''Factor used when converting scalar alms to spin wlms,
    sign*sqrt((l+s)! / (l-s)!)'''

    if spin < 0:
        sign = (-1.0)**spin
    else:
        sign = 1

    spin = np.abs(spin)

    nfact = np.ones_like(ells)
    for i in range(2*spin):
        nfact *= ells + spin - i

    # Since factorial ends at 0!, we would never actually multiply by 0 or a
    # negative number
    nfact[ells < spin] = 0.0

    nfact = sign*np.sqrt(nfact)

    return nfact


def apodizedqu2pureeb(polxscal, polxvect, polxtens, lmax=None, mmax=None):
    '''
    Constructs the pure E/B alms from polarization maps apodized by the
    scalar, vector, and tensor window
    '''

    alm = H.map2alm((-polxtens[0], -polxtens[1]), lmax=lmax, mmax=mmax,
                    pol=False)

    if lmax is None:
        lmax = H.Alm.getlmax(len(alm[0]))

    ell, m_tmp = H.Alm.getlm(lmax)

    idx = (ell >= 2)

    apurelm = (np.zeros_like(alm[0]), np.zeros_like(alm[0]))

    spinfact = 1.0/np.sqrt((ell[idx]-1.0)*ell[idx]*(ell[idx]+1.0)*
                           (ell[idx]+2.0))
    apurelm[0][idx] += spinfact*alm[0][idx]
    apurelm[1][idx] += spinfact*alm[1][idx]

    alm = H.map2alm_spin(polxvect, 1, lmax=lmax, mmax=mmax)
    spinfact = 2.0/np.sqrt((ell[idx]-1.0)*(ell[idx]+2.0))
    apurelm[0][idx] += spinfact*alm[0][idx]
    apurelm[1][idx] += spinfact*alm[1][idx]

    alm = H.map2alm_spin(polxscal, 2, lmax=lmax, mmax=mmax)
    spinfact = 1.0
    apurelm[0][idx] += spinfact*alm[0][idx]
    apurelm[1][idx] += spinfact*alm[1][idx]

    return apurelm


def mapqu2pureeb(mapqu, window, mask=None, lmax=None, mmax=None):
    '''
    Calculates the pure E/B alms from the input Q/U maps and the apodized
    window.
    '''

    nside = H.npix2nside(len(mapqu[0]))

    if mask is None:
        mask = np.ones_like(window)
        mask[window == 0] = 0
    else:
        window = window*mask

    if lmax is None:
        lmax = 3*nside - 1

    if mmax is None:
        mmax = lmax

    nlm = H.Alm.getsize(lmax, mmax=mmax)
    apurelm_eb = (np.zeros(nlm, dtype=np.complex128),
                  np.zeros(nlm, dtype=np.complex128))

    window_scal = (window, np.zeros_like(window))
    window_vect, window_tens = window2vecttens(window, mask=mask)

    polxscal = _apodize_maps(mapqu, window_scal)
    polxvect = _apodize_maps(mapqu, window_vect)
    polxtens = _apodize_maps(mapqu, window_tens)

    apurelm_eb = apodizedqu2pureeb(polxscal, polxvect, polxtens, lmax=lmax,
                                   mmax=mmax)

    return apurelm_eb


def _apodize_maps(mapqu, window):
    '''This is apodizing the input QU maps by multiplying by the complex
    conjugate of the input window.
    '''

    mapqu_ap_0 = mapqu[0]*window[0] + mapqu[1]*window[1]
    mapqu_ap_1 = mapqu[1]*window[0] - mapqu[0]*window[1]

    return mapqu_ap_0, mapqu_ap_1


def map2purealm(maps, window, mask=None, lmax=None, mmax=None, iter=3,
                use_weights=False, datapath=None):
    '''Similar to healpy.map2alm except calculates pure alms
    '''

    nmaps = len(maps)

    if nmaps < 3:
        raise ValueError("""Number of input maps must be 3 (I,Q,U) or 4
                         (I,Q,U,V). Use mapqu2pureeb if you only want
                         to input Q and U maps.""")

    if mask is None:
        mask = np.zeros_like(window)
        mask[window > 0] = 1.0
    else:
        window = window*mask

    alm_i = H.map2alm(maps[0]*window, lmax=lmax, mmax=mmax, iter=iter,
                      use_weights=use_weights, datapath=datapath)

    apurelm_eb = mapqu2pureeb(maps[1:3], window, lmax=lmax, mmax=mmax)

    if nmaps == 4:
        alm_v = H.map2alm(maps[3]*window, lmax=lmax, mmax=mmax, iter=iter,
                          use_weights=use_weights, datapath=datapath)
        return alm_i, apurelm_eb[0], apurelm_eb[1], alm_v
    else:
        return alm_i, apurelm_eb[0], apurelm_eb[1]


def map2alm(maps, lmax=None, mmax=None, iter=3, use_weights=False,
            pol=True, datapath=None):
    '''Extension to H.map2alm which can take in 4 maps (I,Q,U,V).'''

    nmaps = len(maps)
    if pol and nmaps == 4:
        alms_ieb = H.map2alm(maps[0:3], lmax=lmax, mmax=mmax, iter=iter,
                             use_weights=use_weights, pol=pol,
                             datapath=datapath)

        alms_v = H.map2alm(maps[3], lmax=lmax, mmax=mmax, iter=iter,
                           use_weights=use_weights, pol=pol,
                           datapath=datapath)

        alms = alms_ieb + (alms_v,)
    else:
        alms = H.map2alm(maps, lmax=lmax, mmax=mmax, iter=iter,
                         use_weights=use_weights, pol=pol,
                         datapath=datapath)

    return alms


def map2hybridalm(maps, window, mask=None, lmax=None, mmax=None, iter=3,
                  use_weights=False, datapath=None):
    '''Similar to healpy.map2alm except calculates hybrid alms. 
    '''

    nmaps = len(maps)

    if nmaps < 3:
        raise ValueError('''Number of input maps must be 3 (I,Q,U) or 4
                            (I,Q,U,V)''')

    if mask is None:
        mask = np.zeros_like(window)
        mask[window > 0] = 1
    else:
        window = window*mask

    maps_masked = (maps[0]*window, maps[1]*window, maps[2]*window)

    alm_ieb = H.map2alm(maps_masked, lmax=lmax, mmax=mmax, iter=iter,
                        use_weights=use_weights, pol=True, datapath=datapath)

    apurelm_eb = mapqu2pureeb(maps[1:3], window, mask=mask, lmax=lmax,
                              mmax=mmax)

    if nmaps == 4:
        alm_v = H.map2alm(maps[3]*window, lmax=lmax, mmax=mmax, iter=iter,
                          use_weights=use_weights, datapath=datapath)
        return alm_ieb[0], alm_ieb[1], apurelm_eb[1], alm_v
    else:
        return alm_ieb[0], alm_ieb[1], apurelm_eb[1]


def pureanafast(map1, swindow, map2=None, mask=None, nspec=None, lmax=None,
                mmax=None, iter=3, alm=False, use_weights=False,
                datapath=None):
    '''Computes the pure power spectrum of a set of 3 (I,Q,U) or 4 (I,Q,U,V)
    Healpix maps, or the cross-spectrum between two sets of maps if *map2* is
    given.

    Parameters
    ----------
    map1 : float, array-like shape (3, Npix) or (4, Npix)
        A sequence of 3 (or 4) arrays representing I, Q, U, (V) maps
    swindow : float, array-like shape (Npix,)
        A Healpix map containing the window function
    map2 : float, array-like shape (3, Npix) or (4, Npix)
        A sequence of 3 (or 4) arrays representing I, Q, U, (V) maps
    mask : float, array-like shape (Npix,)
        The mask for the maps. If None, generated from the scalar window
    nspec : None or int, optional
        The number of spectra to return. If None, returns all, otherwise
        returns cls[:nspec]
    lmax : int, scalar, optional
        Maximum l of the power spectrum (default: 3*nside-1)
    mmax : int, scalar, optional
        Maximum m of the alm (default: lmax)
    iter : int, scalar, optional
        Number of iteration (default: 3)
    alm : bool, scalar, optional
        If True, returns both cl and alm, otherwise only cl is returned
    datapath : None or str, optional
        If given, the directory where to find the weights data.

    Returns
    -------
    res : array or sequence of arrays
        If *alm* is False, returns cl or a list of cl's (TT, EE, BB, etc.)
        Otherwise, returns a tuple (cl, alm), where cl is as above and
        alm is the spherical harmonic transform or a list of almT, almE, almB
        for polarized input
    '''

    map1 = H.pixelfunc.ma_to_array(map1)
    alms1 = map2purealm(map1, swindow, mask=mask, lmax=lmax, mmax=mmax,
                        iter=iter, use_weights=use_weights, datapath=datapath)
    if map2 is not None:
        map2 = H.pixelfunc.ma_to_array(map2)
        alms2 = map2purealm(map2, swindow, mask=mask, lmax=lmax, mmax=mmax,
                            iter=iter, use_weights=use_weights,
                            datapath=datapath)
    else:
        alms2 = None

    cls = H.alm2cl(alms1, alms2=alms2, lmax=lmax, mmax=mmax, lmax_out=lmax,
                   nspec=nspec)

    if alm:
        if map2 is not None:
            return (cls, alms1, alms2)
        else:
            return (cls, alms1)
    else:
        return cls


def hybridanafast(map1, swindow, map2=None, mask=None, nspec=None, lmax=None,
                  mmax=None, iter=3, alm=False, use_weights=False,
                  datapath=None):
    '''Computes the hybrid power spectrum of a set of 3 (I,Q,U) or 4 (I,Q,U,V)
    Healpix maps, or the cross-spectrum between two sets of maps if *map2* is
    given. Hybrid power spectra are ones where E is pseudo and B is pure.

    Parameters
    ----------
    map1 : float, array-like shape (3, Npix) or (4, Npix)
        A sequence of 3 (or 4) arrays representing I, Q, U, (V) maps
    swindow : float, array-like shape (Npix,)
        A Healpix map containing the window function
    map2 : float, array-like shape (3, Npix) or (4, Npix)
        A sequence of 3 (or 4) arrays representing I, Q, U, (V) maps
    mask : float, array-like shape (Npix,)
        The mask for the maps. If None, generated from the scalar window
    nspec : None or int, optional
        The number of spectra to return. If None, returns all, otherwise
        returns cls[:nspec]
    lmax : int, scalar, optional
        Maximum l of the power spectrum (default: 3*nside-1)
    mmax : int, scalar, optional
        Maximum m of the alm (default: lmax)
    iter : int, scalar, optional
        Number of iteration (default: 3)
    alm : bool, scalar, optional
        If True, returns both cl and alm, otherwise only cl is returned
    datapath : None or str, optional
        If given, the directory where to find the weights data.

    Returns
    -------
    res : array or sequence of arrays
        If *alm* is False, returns cl or a list of cl's (TT, EE, BB, etc.)
        Otherwise, returns a tuple (cl, alm), where cl is as above and
        alm is the spherical harmonic transform or a list of almT, almE, almB,
        (almV)
    '''

    map1 = H.pixelfunc.ma_to_array(map1)
    alms1 = map2hybridalm(map1, swindow, mask=mask, lmax=lmax, mmax=mmax,
                          iter=iter, use_weights=use_weights,
                          datapath=datapath)
    if map2 is not None:
        map2 = H.pixelfunc.ma_to_array(map2)
        alms2 = map2hybridalm(map2, swindow, mask=mask, lmax=lmax, mmax=mmax,
                              iter=iter, use_weights=use_weights,
                              datapath=datapath)
    else:
        alms2 = None

    cls = H.alm2cl(alms1, alms2=alms2, lmax=lmax, mmax=mmax, lmax_out=lmax,
                   nspec=nspec)

    if alm:
        if map2 is not None:
            return (cls, alms1, alms2)
        else:
            return (cls, alms1)
    else:
        return cls


def anafast(map1, map2=None, nspec=None, lmax=None, mmax=None, iter=3,
            alm=False, pol=True, use_weights=False, datapath=None):
    '''Computes the power spectrum of an Healpix map, or the cross-spectrum
    between two maps if *map2* is given.
    No removal of monopole or dipole is performed. This is an extension of
    healpy.anafast for a possibility of 4 Stokes parameters

    Parameters
    ----------
    map1 : float, array-like shape (Npix,) or (3, Npix) or (4, Npix)
      Either an array representing a map, or a sequence of
      3 (I,Q,U), or 4 (I,Q,U,V) arrays
    map2 : float, array-like shape (Npix,) or (3, Npix) or (4, Npix)
      Either an array representing a map, or a sequence of
      3 (I,Q,U), or 4 (I,Q,U,V) arrays
    nspec : None or int, optional
      The number of spectra to return. If None, returns all, otherwise
      returns cls[:nspec]
    lmax : int, scalar, optional
      Maximum l of the power spectrum (default: 3*nside-1)
    mmax : int, scalar, optional
      Maximum m of the alm (default: lmax)
    iter : int, scalar, optional
      Number of iteration (default: 3)
    alm : bool, scalar, optional
      If True, returns both cl and alm, otherwise only cl is returned
    pol : bool, optional
      If True, assumes input maps are TQU. Output will be TEB cl's and
      correlations (input must be 1 or 3 maps).
      If False, maps are assumed to be described by spin 0 spherical harmonics
      (input can be any number of maps)
      If there is only one input map, it has no effect. Default: True.
    datapath : None or str, optional
      If given, the directory where to find the weights data.

    Returns
    -------
    res : array or sequence of arrays
      If *alm* is False, returns cl or a list of cl's (TT, EE, BB, TE, EB, TB
      for polarized input map)
      Otherwise, returns a tuple (cl, alm), where cl is as above and
      alm is the spherical harmonic transform or a list of almT, almE, almB
      for polarized input
    '''

    nmaps = len(map1)
    if nmaps > 10:
        nmaps = 1

    if pol and (nmaps < 3):
        raise ValueError('If pol=True, then nmaps must be at least 3 (I,Q,U)')

    map1 = H.pixelfunc.ma_to_array(map1)
    alms1 = H.map2alm(map1[:3], lmax=lmax, mmax=mmax, iter=iter, pol=pol,
                      use_weights=use_weights, datapath=datapath)
    if nmaps == 4:
        alms1_v = H.map2alm(map1[3], lmax=lmax, mmax=mmax, iter=iter,
                            use_weights=use_weights, datapath=datapath)
        alms1 = alms1 + (alms1_v, )

    if map2 is not None:
        map2 = H.pixelfunc.ma_to_array(map2)
        alms2 = H.map2alm(map2[:3], lmax=lmax, mmax=mmax, iter=iter, pol=pol,
                          use_weights=use_weights, datapath=datapath)
        if nmaps == 4:
            alms2_v = H.map2alm(map2[3], lmax=lmax, mmax=mmax, iter=iter,
                                use_weights=use_weights, datapath=datapath)
            alms2 = alms2 + (alms2_v, )
    else:
        alms2 = None

    cls = H.alm2cl(alms1, alms2=alms2, lmax=lmax, mmax=mmax, lmax_out=lmax,
                   nspec=nspec)

    if alm:
        if map2 is not None:
            return (cls, alms1, alms2)
        else:
            return (cls, alms1)
    else:
        return cls
