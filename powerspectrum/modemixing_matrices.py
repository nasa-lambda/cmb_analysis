# pylint: disable=E1101, C0103, R0912, R0913, R0914, R0915, W0212

#Copyright 2016 United States Government as represented by the Administrator
#of the National Aeronautics and Space Administration. All Rights Reserved.

'''
Calculation of modemixing matrices using Python application of these
matrices to Cls. Parallelization is done using MPI.
'''

import numpy as np
import healpy as H

import cmb_analysis.powerspectrum.healpy_ext as H_ext
import cmb_analysis.util.wignercoupling as wc


def apply_inverse_coupling(cls_in, Mscal, Mpol, Mcross):
    '''
    Solve for the true Cls from the pseudo/pure Cls
    
    Parameters
    ----------
    cls_in : array-like (ncls, nell)
        Input Cls, either 6 (T, E, B) or 10 (T, E, B, V)

    Mscal : array-like (nell, nell)
        Mode mixing matrix used for TT and VV

    Mpol: array-like (3*nell, 3*nell)
        Mode mixing matrix for EE, BB, EB mixing

    Mcross: array-like (2*nell, 2*nell)
        Mode mixing matrix for TE, TB or EV, BV mixing

    Notes
    -----
    This gets the true Cls by essentially solving the equation Ax=b by
    least squares procedure.
    '''

    ncls = len(cls_in)
    
    if Mscal is not None:
        nell = Mscal.shape[0]
    elif Mpol is not None:
        nell = Mpol.shape[0] / 3
    elif Mcross is not None:
        nell = Mcross.shape[0] / 2
    else:
        raise ValueError("One of Mscal, Mpol, and Mcross needs to be input")

    cls_out = np.empty([ncls, nell])
    
    if ncls == 10:
        TT, EE, BB, VV, TE, TB, TV, EB, EV, BV = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        doV = True
    elif ncls == 6:
        TT, EE, BB, TE, EB, TB = 0, 1, 2, 3, 4, 5
        doV = False
    else:
        raise ValueError('''Input cls need to have 6 (without V) or 10
                            (with V) power spectra''')

    if Mscal is None:
        Mscal = np.zeros([nell, nell])

    if Mpol is None:
        Mpol = np.zeros([nell, nell])

    if Mcross is None:
        Mcross = np.zeros([nell, nell])

    func = np.linalg.lstsq

    if Mscal is not None:
        cls_out[TT, :] = func(Mscal, cls_in[TT])[0]

        if doV:
            cls_out[VV, :] = func(Mscal, cls_in[VV])[0]
            cls_out[TV, :] = func(Mscal, cls_in[TV])[0]

    if Mpol is not None:
        polvect = np.empty(3*nell)
        polvect[0:nell] = cls_in[EE]
        polvect[nell:2*nell] = cls_in[BB]
        polvect[2*nell:] = cls_in[EB]

        polvect_out = func(Mpol, polvect)
        polvect_out = polvect_out[0]

        cls_out[EE, :] = polvect_out[0:nell]
        cls_out[BB, :] = polvect_out[nell:2*nell]
        cls_out[EB, :] = polvect_out[2*nell:]

    if Mcross is not None:
        crossvect = np.empty(2*nell)
        crossvect[0:nell] = cls_in[TE]
        crossvect[nell:2*nell] = cls_in[TB]

        crossvect_out = func(Mcross, crossvect)
        crossvect_out = crossvect_out[0]

        cls_out[TE, :] = crossvect_out[0:nell]
        cls_out[TB, :] = crossvect_out[nell:2*nell]

        if doV:
            crossvect[0:nell] = cls_in[EV]
            crossvect[nell:2*nell] = cls_in[BV]

            crossvect_out = func(Mcross, crossvect)
            crossvect_out = crossvect_out[0]

            cls_out[EV, :] = crossvect_out[0:nell]
            cls_out[BV, :] = crossvect_out[nell:2*nell]

    return cls_out

def apply_coupling(cls_in, Mscal, Mpol, Mcross, inverse=True):
    '''Solve for estimates of true Cls given input pseudo/pure Cls and
    mode-mixing matrices or apply the mode-mixing matrices to full sky
    Cls to get estimates of te pure/pseudo Cls.

    Parameters
    ----------
    cls_in : array-like (ncls, nell)
        Input Cls, either 6 (T, E, B) or 10 (T, E, B, V)

    Mscal : array-like (nell, nell)
        Mode mixing matrix used for TT and VV

    Mpol: array-like (3*nell, 3*nell)
        Mode mixing matrix for EE, BB, EB mixing

    Mcross: array-like (2*nell, 2*nell)
        Mode mixing matrix for TE, TB or EV, BV mixing

    inverse: boolean, optional (default: True)
        Whether to solve for the full sky Cls given in pseudo/pure Cls or
        calculate the pseudo/pure given the full sky Cls
        

    Notes
    -----
    For solving the inverse equation, we calculate the pseudo-inverse of 
    the matrix (np.linalg.pinv).
    '''

    ncls = len(cls_in)

    if Mscal is not None:
        nell = Mscal.shape[0]
    elif Mpol is not None:
        nell = Mpol.shape[0] / 3
    elif Mcross is not None:
        nell = Mcross.shape[0] / 2
    else:
        raise ValueError("One is Mscal, Mpol, and Mcross needs to be input")

    if ncls == 10:
        TT, EE, BB, VV, TE, TB, TV, EB, EV, BV = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        doV = True
    elif ncls == 6:
        TT, EE, BB, TE, EB, TB = 0, 1, 2, 3, 4, 5
        doV = False
    elif ncls == 4:
        doV = False
        TT, EE, BB, TE = 0, 1, 2, 3

    cls_out = np.empty([ncls, nell])

    if Mscal is None:
        Mscal = np.zeros([nell, nell])

    if Mpol is None:
        Mpol = np.zeros([nell, nell])

    if Mcross is None:
        Mcross = np.zeros([nell, nell])

    if inverse:
        B_scal = np.linalg.pinv(Mscal, rcond=1e-10)
        B_pol = np.linalg.pinv(Mpol, rcond=1e-10)
        B_cross = np.linalg.pinv(Mcross, rcond=1e-10)
    else:
        B_scal = Mscal
        B_pol = Mpol
        B_cross = Mcross

    cls_out[TT, :] = np.dot(B_scal, cls_in[TT])

    if doV:
        cls_out[VV, :] = np.dot(B_scal, cls_in[VV])
        cls_out[TV, :] = np.dot(B_scal, cls_in[TV])

    polvect = np.zeros(3*nell)
    polvect[0:nell] = cls_in[EE]
    polvect[nell:2*nell] = cls_in[BB]
    if ncls > 4:
        polvect[2*nell:] = cls_in[EB]

    polvect_out = np.dot(B_pol, polvect)

    cls_out[EE, :] = polvect_out[0:nell]
    cls_out[BB, :] = polvect_out[nell:2*nell]

    if ncls > 4:
        cls_out[EB, :] = polvect_out[2*nell:]

    crossvect = np.zeros(2*nell)
    crossvect[:nell] = cls_in[TE]
    if ncls > 4:
        crossvect[nell:] = cls_in[TB]

    crossvect_out = np.dot(B_cross, crossvect)

    cls_out[TE, :] = crossvect_out[:nell]

    if ncls > 4:
        cls_out[TB, :] = crossvect_out[nell:]

    if doV:
        crossvect[:nell] = cls_in[EV]
        crossvect[nell:] = cls_in[BV]

        crossvect_out = np.dot(B_cross, crossvect)

        cls_out[EV, :] = crossvect_out[:nell]
        cls_out[BV, :] = crossvect_out[:nell]

    return cls_out


def _get_wEBlm(window_scal, lmax=None, maps=False):
    '''
    Calculate the E/B-type alms from the vector and tensor windows.
    Constructs the vector and tensor windows from the scalar wlms and
    then masks the resulting maps. This masking results in non-zero B-mode
    type wlms.
    '''

    mask = np.ones_like(window_scal)
    mask[window_scal == 0] = 0

    window_vect, window_tens = H_ext.window2vecttens(window_scal, mask=mask,
                                                     lmax=lmax)

    wlm = H.map2alm(window_scal, lmax=lmax)

    n = len(wlm)
    wEBlm = np.empty([5, n], dtype=np.complex)
    wEBlm[0, :] = -wlm

    wlm_tmp = H.map2alm_spin(window_vect, 1, lmax=lmax)
    wEBlm[1, :] = wlm_tmp[0]
    wEBlm[-1, :] = wlm_tmp[1]

    wlm_tmp = H.map2alm_spin(window_tens, 2, lmax=lmax)
    wEBlm[2, :] = wlm_tmp[0]
    wEBlm[-2, :] = wlm_tmp[1]

    if maps:
        window_vect = window_vect[0] + 1j*window_vect[1]
        window_tens = window_tens[0] + 1j*window_tens[1]
        return wEBlm, (window_vect, window_tens)
    else:
        return wEBlm


def calc_modemixing(comm, window, cl_type='pseudo', lmax=None, scal=True,
                    pol=True, cross=True, verbose=True):
    '''
    Calculates the mode-mixing matrices given an input full sky window in
    Healpix format

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (normally COMM_WORLD)

    window : array-like
        Healpix scalar window

    cl_type : 'pseudo', 'pure', or 'hybrid'
        The type of Cl that we want the mode-mixing matrices

    lmax : int, scalar, optional
        Maximum l of the power spectrum. Default: 3*nside-1

    scal : bool, optional
        Whether to calculate the mode-mixing matrix for temperature

    pol : bool, optional
        Whether to calculate the mode-mixing matrix for polarization

    cross : bool, optional
        Whether to calculate the mode-mixing matrix for temp-pol

    Returns
    -------
    Mout : list
        A list of the mode-mixing matrices in order: Mscal, Mpol, Mcross
        Only requested matrices are calculated and returned.
    '''

    if lmax is None:
        nside = H.npix2nside(len(window))
        lmax = 3*nside - 1

    wEBlm = _get_wEBlm(window, lmax=lmax)

    Mout = []

    if scal:
        Mscal = _calc_Mscal(comm, wEBlm, lmax=lmax, verbose=verbose)
        Mout.append(Mscal)

    if pol:
        Mpol = _calc_Mpol(comm, wEBlm, lmax=lmax, cl_type=cl_type,
                          verbose=verbose)
        Mout.append(Mpol)

    if cross:
        Mcross = _calc_Mcross(comm, wEBlm, lmax=lmax, cl_type=cl_type,
                              verbose=verbose)
        Mout.append(Mcross)

    return Mout


def _calc_Mscal(comm, wEBlm, lmax=None, verbose=True):
    '''Calculate the temperature/V polarization mode-mixing matrix
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()

    if lmax is None:
        lmax = H.Alm.getlmax(len(wEBlm[0, :]))

    Mscal = np.zeros([lmax+1, lmax+1])
    wl = H.alm2cl(wEBlm[0, :])

    for l1 in range(2+rank, lmax+1, size):
        if verbose:
            print("Scal l1 = ", l1)
        for l2 in range(2, lmax+1):
            l3min = np.abs(l1-l2)
            l3max = np.abs(l1+l2)

            l3vals = np.arange(l3min, l3max+1)

            # Calculates Wigner 3j symbol for all valid l3
            JT = wc.wigner3j_vect(2*l1, 2*l2, 0, 0)

            # Since we only have wl up to lmax we must ignore all terms that
            # have l3 > lmax
            idx = l3vals <= lmax

            Mscal[l1, l2] = np.sum((2*l3vals[idx]+1)*wl[l3vals[idx]] *
                                   JT[idx]**2)

            normfact = (2.0*l2+1.0) / (4.0*np.pi)
            Mscal[l1, l2] *= normfact

    Mscal = comm.allreduce(Mscal)

    return Mscal


def _calc_Mpol(comm, wEBlm, lmax=None, cl_type='pseudo', verbose=True):
    '''Calculate the polarization mode mixing matrix
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()

    if lmax is None:
        lmax = H.Alm.getlmax(len(wEBlm[0, :]))
    
    l, m = H.Alm.getlm(lmax)

    Mpol = np.zeros([3*lmax+3, 3*lmax+3])

    a0 = 1.0
    a1 = 2.0
    a2 = 1.0

    l1_vals = range(2+rank, lmax+1, size)

    # Don't need to calculate this if we are looking at pseudo Cls, but it
    # is not the bottleneck in the calculation so I don't care
    Nl12 = H_ext._nfunc(l1_vals, 2)
    Nl11 = H_ext._nfunc(l1_vals, 1)
    Nl10 = H_ext._nfunc(l1_vals, 0)
    fact0s = Nl10 / Nl12
    fact1s = Nl11 / Nl12

    for l1, fact0, fact1 in zip(l1_vals, fact0s, fact1s):
        if verbose:
            print("Pol l1 = ", l1)

        for l2 in range(2, lmax+1):
            l3min = np.abs(l1-l2)
            l3max = np.abs(l1+l2)

            # Minimum l3 for when m3 is 1 or 2
            l3min2 = np.max([l3min, 2])
            l3min1 = np.max([l3min, 1])

            # Wigner Symbols that we need are
            # (l1   l2 l3 ) and (l1   l2 l3) for m=0 (pseudo)
            # (-2+m 2  -m )     (2-m  -2 m ) and m=1,2 (pure)

            # m=0 term needed for pseudo and pure
            wc_0 = wc.wigner3j_vect(2*l1, 2*l2, -2*2, 2*2)
            wc_1 = wc.wigner3j_vect(2*l1, 2*l2, 2*2, -2*2)
            Jp0 = wc_0 + wc_1
            Jm0 = wc_0 - wc_1

            # m=1,2 terms are only needed for pure modes
            if (cl_type == 'pure') or (cl_type == 'hybrid'):
                wc_0 = wc.wigner3j_vect(2*l1, 2*l2, (-2+1)*2, 2*2)
                wc_1 = wc.wigner3j_vect(2*l1, 2*l2, (2-1)*2, -2*2)
                Jp1 = wc_0 + wc_1
                Jm1 = wc_0 - wc_1

                wc_0 = wc.wigner3j_vect(2*l1, 2*l2, (-2+2)*2, 2*2)
                wc_1 = wc.wigner3j_vect(2*l1, 2*l2, (2-2)*2, -2*2)
                Jp2 = wc_0 + wc_1
                Jm2 = wc_0 - wc_1

            # This allows us to remove any sum over m or l3
            idx = np.all([l >= l3min, l <= l3max], axis=0)
            l_tmp = l[idx]
            m_tmp = m[idx]
            m0 = m_tmp == 0

            wEBlm_tmp = wEBlm[:, idx]

            # Calculate the correct index in the Jp and Jm terms
            idx_l3_0 = np.array(l_tmp - l3min, dtype=np.int)  # Jp0, Jm0
            idx_l3_1 = np.array(l_tmp - l3min1, dtype=np.int)  # Jp1, Jm1
            idx_l3_2 = np.array(l_tmp - l3min2, dtype=np.int)  # Jp2, Jm2

            # When l3min is 0 or 1 and take the whole range, Jp1(Jm1) and/or
            # Jp2(Jm2) get non-zero values when they should be zero (when l3 =
            # 0 or 1) because negative indices in idx_l3_1 and idx_l3_2 wrap
            # around to the end
            idx1 = l_tmp >= l3min1
            idx2 = l_tmp >= l3min2

            # EE,EE and BB,BB (2/9)
            termE = a2 * wEBlm_tmp[0, :] * Jp0[idx_l3_0]
            termB = np.zeros_like(termE)

            if cl_type == 'pseudo' or cl_type == 'hybrid':
                Mpol[l1, l2] = 2*np.sum(termE*np.conj(termE))
                Mpol[l1, l2] -= np.sum(termE[m0]*np.conj(termE[m0]))

            if cl_type == 'pure' or cl_type == 'hybrid':
                termE[idx1] += a1*fact1*wEBlm_tmp[1, idx1]*Jp1[idx_l3_1][idx1]
                termE[idx2] += a0*fact0*wEBlm_tmp[2, idx2]*Jp2[idx_l3_2][idx2]
                termB[idx1] += a1*fact1*wEBlm_tmp[-1, idx1]*Jm1[idx_l3_1][idx1]
                termB[idx2] += a0*fact0*wEBlm_tmp[-2, idx2]*Jm2[idx_l3_2][idx2]

            Mpol[l1+lmax+1, l2+lmax+1] = 2*np.sum(termE*np.conj(termE) +
                                                  termB*np.conj(termB))
            Mpol[l1+lmax+1, l2+lmax+1] -= np.sum(termE[m0]*np.conj(termE[m0]) +
                                                 termB[m0]*np.conj(termB[m0]))

            if cl_type == 'pure':
                Mpol[l1, l2] = Mpol[l1+lmax+1, l2+lmax+1]

            # EE,BB and BB,EE (4/9)
            termE = a2 * wEBlm_tmp[0, :] * Jm0[idx_l3_0]
            termB = np.zeros_like(termE)

            if cl_type == 'pseudo' or cl_type == 'hybrid':
                Mpol[l1, l2+lmax+1] = 2*np.sum(termE*np.conj(termE))
                Mpol[l1, l2+lmax+1] -= np.sum(termE[m0]*np.conj(termE[m0]))

            if cl_type == 'pseudo':
                Mpol[l1+lmax+1, l2] = Mpol[l1, l2+lmax+1]

            if cl_type == 'pure' or cl_type == 'hybrid':
                termE[idx1] += a1*fact1*wEBlm_tmp[1, idx1]*Jm1[idx_l3_1][idx1]
                termE[idx2] += a0*fact0*wEBlm_tmp[2, idx2]*Jm2[idx_l3_2][idx2]
                termB[idx1] += a1*fact1*wEBlm_tmp[-1, idx1]*Jp1[idx_l3_1][idx1]
                termB[idx2] += a0*fact0*wEBlm_tmp[-2, idx2]*Jp2[idx_l3_2][idx2]

                Mpol[l1+lmax+1, l2] = 2*np.sum(termE*np.conj(termE) +
                                               termB*np.conj(termB))
                Mpol[l1+lmax+1, l2] -= np.sum(termE[m0]*np.conj(termE[m0]) +
                                              termB[m0]*np.conj(termB[m0]))

            if cl_type == 'pure':
                Mpol[l1, l2+lmax+1] = Mpol[l1+lmax+1, l2]

            # EB,EB (5/9)
            if cl_type == 'pseudo' or cl_type == 'pure':
                # EE,EE - EE,BB
                Mpol[l1+2*lmax+2, l2+2*lmax+2] = Mpol[l1, l2] - Mpol[l1, l2+lmax+1]
            else:
                termE = np.zeros_like(Jp0[idx_l3_0], dtype=np.float64)
                termE[idx2] += a0*fact0*(Jp0[idx_l3_0]*Jp2[idx_l3_2]-Jm0[idx_l3_0]*Jm2[idx_l3_2])[idx2] * np.real(wEBlm_tmp[0, idx2]*np.conj(wEBlm_tmp[2, idx2]))
                termE[idx1] += a1*fact1*(Jp0[idx_l3_0]*Jp1[idx_l3_1]-Jm0[idx_l3_0]*Jm1[idx_l3_1])[idx1] * np.real(wEBlm_tmp[0, idx1]*np.conj(wEBlm_tmp[1, idx1]))
                termE += a2 * (Jp0[idx_l3_0]*Jp0[idx_l3_0] - Jm0[idx_l3_0]*Jm0[idx_l3_0]) * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[0, :]))
                Mpol[l1+2*lmax+2, l2+2*lmax+2] = 2*np.sum(termE) - np.sum(termE[m0])

            # EE,EB and BB,EB (7/9)
            if cl_type == 'pure' or cl_type == 'hybrid':
                termE = np.zeros_like(Jp0[idx_l3_0], dtype=np.float64)
                # only non-zero for pure (and B is pure in hybrid).
                termE[idx2] += a0*a0*fact0*fact0*(Jp2[idx_l3_2]*Jp2[idx_l3_2]*np.real(wEBlm_tmp[2, :]*np.conj(wEBlm_tmp[-2, :]))
                                                  - Jm2[idx_l3_2]*Jm2[idx_l3_2]*np.real(wEBlm_tmp[-2, :]*np.conj(wEBlm_tmp[2, :])))[idx2]
                termE[idx2] += a0*a1*fact0*fact1*(Jp2[idx_l3_2]*Jp1[idx_l3_1]*np.real(wEBlm_tmp[2, :]*np.conj(wEBlm_tmp[-1, :]))
                                                  - Jm2[idx_l3_2]*Jm1[idx_l3_1]*np.real(wEBlm_tmp[-2, :]*np.conj(wEBlm_tmp[1, :])))[idx2]
                termE[idx2] += a0*a2*fact0*(-Jm2[idx_l3_2]*Jm0[idx_l3_0]*np.real(wEBlm_tmp[-2, :]*np.conj(wEBlm_tmp[0, :])))[idx2]
                termE[idx2] += a1*a0*fact1*fact0*(Jp1[idx_l3_1]*Jp2[idx_l3_2]*np.real(wEBlm_tmp[1, :]*np.conj(wEBlm_tmp[-2, :]))
                                                  - Jm1[idx_l3_1]*Jm2[idx_l3_2]*np.real(wEBlm_tmp[-1, :]*np.conj(wEBlm_tmp[2, :])))[idx2]
                termE[idx1] += a1*a1*fact1*fact1*(Jp1[idx_l3_1]*Jp1[idx_l3_1]*np.real(wEBlm_tmp[1, :]*np.conj(wEBlm_tmp[-1, :]))
                                                  - Jm1[idx_l3_1]*Jm1[idx_l3_1]*np.real(wEBlm_tmp[-1, :]*np.conj(wEBlm_tmp[1, :])))[idx1]
                termE[idx1] += a1*a2*fact1*(-Jm1[idx_l3_1]*Jm0[idx_l3_0]*np.real(wEBlm_tmp[-1, :]*np.conj(wEBlm_tmp[0, :])))[idx1]
                termE[idx2] += a2*a0*fact0*(Jp0[idx_l3_0]*Jp2[idx_l3_2]*np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-2, :])))[idx2]
                termE[idx1] += a2*a1*fact1*(Jp0[idx_l3_0]*Jp1[idx_l3_1]*np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-1, :])))[idx1]

                Mpol[l1+lmax+1, l2+2*lmax+2] = 2*np.sum(termE) - np.sum(termE[m0])

            if cl_type == 'pure':
                Mpol[l1, l2+2*lmax+2] = Mpol[l1+lmax+1, l2+2*lmax+2]

            # EB,EE and EB,BB (9/9)
            if cl_type == 'pure':
                Mpol[l1+2*lmax+2, l2] = Mpol[l1, l2+2*lmax+2]
                Mpol[l1+2*lmax+2, l2+lmax+1] = Mpol[l1, l2+2*lmax+2]
            elif cl_type == 'hybrid':
                termE = np.zeros_like(Jp0[idx_l3_0], dtype=np.float64)
                termE[idx2] += a0*fact0*(Jp0[idx_l3_0]*Jp2[idx_l3_2]*np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-2, :])))[idx2]
                termE[idx1] += a1*fact1*(Jp0[idx_l3_0]*Jp1[idx_l3_1]*np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-1, :])))[idx1]

                termB = np.zeros_like(Jm0[idx_l3_0], dtype=np.float64)
                termB[idx2] += a0*fact0*(Jm0[idx_l3_0]*Jm2[idx_l3_2]*np.real(wEBlm_tmp[-2, :]*np.conj(wEBlm_tmp[0, :])))[idx2]
                termB[idx1] += a1*fact1*(Jm0[idx_l3_0]*Jm1[idx_l3_1]*np.real(wEBlm_tmp[-1, :]*np.conj(wEBlm_tmp[0, :])))[idx1]

                Mpol[l1+2*lmax+2, l2] = 2*np.sum(termE) - np.sum(termE[m0])
                Mpol[l1+2*lmax+2, l2+lmax+1] = 2*np.sum(termB) - np.sum(termB[m0])

            normfact = (2.0*l2+1.0) / (4.0*np.pi)
            Mpol[l1, l2] *= normfact / 4.0
            Mpol[l1+lmax+1, l2+lmax+1] *= normfact / 4.0
            Mpol[l1, l2+lmax+1] *= normfact / 4.0
            Mpol[l1+lmax+1, l2] *= normfact / 4.0
            Mpol[l1+2*lmax+2, l2+2*lmax+2] *= normfact / 4.0
            Mpol[l1, l2+2*lmax+2] *= normfact / 2.0
            Mpol[l1+lmax+1, l2+2*lmax+2] *= -normfact / 2.0
            Mpol[l1+2*lmax+2, l2] *= -normfact / 4.0
            Mpol[l1+2*lmax+2, l2+lmax+1] *= normfact / 4.0

    Mpol = comm.allreduce(Mpol)

    return Mpol


def _calc_Mcross(comm, wEBlm, lmax=None, cl_type='pseudo', verbose=True):
    '''Calculate the temp-pol mode-mixing matrix.
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()

    if lmax is None:
        lmax = H.Alm.getlmax(len(wEBlm[0, :]))
    
    l, m = H.Alm.getlm(lmax)

    Mcross = np.zeros([2*lmax+2, 2*lmax+2])

    a0 = 1.0
    a1 = 2.0
    a2 = 1.0

    l1_vals = range(2+rank, lmax+1, size)

    # Don't need to calculate this if we are looking at pseudo Cls, but it
    # is not the bottleneck in the calculation so I don't care
    Nl12 = H_ext._nfunc(l1_vals, 2)
    Nl11 = H_ext._nfunc(l1_vals, 1)
    Nl10 = H_ext._nfunc(l1_vals, 0)
    fact0s = Nl10 / Nl12
    fact1s = Nl11 / Nl12

    for l1, fact0, fact1 in zip(l1_vals, fact0s, fact1s):
        if verbose:
            print("Cross l1 = ", l1)

        for l2 in range(2, lmax+1):
            l3min = np.abs(l1-l2)
            l3max = np.abs(l1+l2)

            # Minimum l3 for case when m3=1,2
            l3min2 = np.max([l3min, 2])
            l3min1 = np.max([l3min, 1])

            wc_0 = wc.wigner3j_vect(2*l1, 2*l2, -2*2, 2*2)
            wc_1 = wc.wigner3j_vect(2*l1, 2*l2, 2*2, -2*2)
            Jp0 = wc_0 + wc_1
#           Jm0 = wc_0 - wc_1
            JT = wc.wigner3j_vect(2*l1, 2*l2, 0, 0)

            if cl_type == 'pure' or cl_type == 'hybrid':
                wc_0 = wc.wigner3j_vect(2*l1, 2*l2, (-2+1)*2, 2*2)
                wc_1 = wc.wigner3j_vect(2*l1, 2*l2, (2-1)*2, -2*2)
                Jp1 = wc_0 + wc_1
#               Jm1 = wc_0 - wc_1

                wc_0 = wc.wigner3j_vect(2*l1, 2*l2, (-2+2)*2, 2*2)
                wc_1 = wc.wigner3j_vect(2*l1, 2*l2, (2-2)*2, -2*2)
                Jp2 = wc_0 + wc_1
#               Jm2 = wc_0 - wc_1

            idx = np.all([l >= l3min, l <= l3max], axis=0)
            l_tmp = l[idx]
            m_tmp = m[idx]
            m0 = m_tmp == 0

            # sub array of wEBlm that have valid l (for l = l3)
            wEBlm_tmp = wEBlm[:, idx]

            idx_l3_0 = np.array(l_tmp - l3min, dtype=np.int)  # Jp0,Jm0
            idx_l3_1 = np.array(l_tmp - l3min1, dtype=np.int)  # Jp1,Jm1
            idx_l3_2 = np.array(l_tmp - l3min2, dtype=np.int)  # Jp2,Jm2

            # Subsets of the subset. When l3min is 0 or 1 and take the whole
            # range, Jp1(Jm1) and/or Jp2(Jm2) get non-zero values when
            # they should be zero (when l3 = 0 or 1) because negative indices
            # in idx_l3_1 and idx_l3_2 wrap around to the end
            idx1 = l_tmp >= l3min1
            idx2 = l_tmp >= l3min2

            # TE,TE, and TB,TB (2/4)
            termE = a2 * JT[idx_l3_0]*Jp0[idx_l3_0] * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[0, :]))

            if cl_type == 'hybrid' or cl_type == 'pseudo':
                Mcross[l1, l2] = 2*np.sum(termE) - np.sum(termE[m0])

            if cl_type == 'pure' or cl_type == 'hybrid':
                termE[idx1] += a1*fact1 * (JT[idx_l3_0]*Jp1[idx_l3_1] * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[1, :])))[idx1]
                termE[idx2] += a0*fact0 * (JT[idx_l3_0]*Jp2[idx_l3_2] * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[2, :])))[idx2]

            Mcross[l1+lmax+1, l2+lmax+1] = 2*np.sum(termE) - np.sum(termE[m0])

            if cl_type == 'pure':
                Mcross[l1, l2] = Mcross[l1+lmax+1, l2+lmax+1]

            # TE,TB and TB,TE (4/4)
            if cl_type == 'pure' or cl_type == 'hybrid':
                termE = np.zeros_like(JT[idx_l3_0], dtype=np.float64)
                termE[idx1] += a1*fact1 * (JT[idx_l3_0]*Jp1[idx_l3_1] * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-1, :])))[idx1]
                termE[idx2] += a0*fact0 * (JT[idx_l3_0]*Jp2[idx_l3_2] * np.real(wEBlm_tmp[0, :]*np.conj(wEBlm_tmp[-2, :])))[idx2]
                Mcross[l1+lmax+1, l2] = 2*np.sum(termE) - np.sum(termE[m0])

            if cl_type == 'pure':
                Mcross[l1, l2+lmax+1] = Mcross[l1+lmax+1, l2]

            normfact = (2.0*l2+1.0) / (4.0*np.pi)

            Mcross[l1, l2] *= normfact / 2.0
            Mcross[l1+lmax+1, l2+lmax+1] *= normfact / 2.0
            Mcross[l1, l2+lmax+1] *= normfact / 2.0
            Mcross[l1+lmax+1, l2] *= -normfact / 2.0

    Mcross = comm.allreduce(Mcross)

    return Mcross
