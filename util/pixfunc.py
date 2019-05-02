# pylint: disable=E1101, C0103, R0912, R0913, R0914, R0915, W0212

#Copyright 2016 United States Government as represented by the Administrator
#of the National Aeronautics and Space Administration. All Rights Reserved.

'''This module provides functions to convert between sky coordinates and
COBE pixel numbers. This is adapted from the IDL code written in support
of COBE data analysis.
'''

from __future__ import print_function, division

import numpy as np
from astropy.coordinates import SkyCoord
import scipy.sparse

def coord2pix(lon, lat=None, coord='C', res='F'):
    '''
    Get the COBE pixel closest to the input longitude/latitude

    Parameters
    ----------
    lon : float, array-like shape (Npts,)
        either longitude (ra) or SkyCoord object (or array of these)
    lat : float, array-like shape (Npts,), optional
        latitude (dec) or array of latitudes. if present, lon is longitude
    coord : string, optional
        coordinate system of the input longitude, latitude. Default is celestial coordinates
        (icrs). Ignored if the input coordinates are SkyCoord objects
    res : string or int, optional
        COBE pixelization resolution. 'F' (FIRAS) = 'D' (DMR) = 6, 'B' (DIRBE) = 9.
        Default is 6. Resolution <= 15

    Returns
    -------
    pix : int, array-like shape (Npts,)
        output pixel number or array of pixel numbers in COBE pixelization
    '''

    if res == 'F' or res == 'D':
        res = 6
    elif res == 'B':
        res = 9


    #I believe this is the relevant code from coorconv.pro that I need to follow
    #uvec = cconv(input*1.,-1) #convert to unit vector
    #uvec = skyconv(uvec, inco=in_coord, outco='E') #convert to ecliptic
    #output = cconv(uvec,out_res) #uv -> pix through ux2pix.pro routine

    npts = np.size(lon)

    if coord == 'C':
        coord = 'icrs'
    elif coord == 'G':
        coord = 'galactic'
    elif coord == 'E':
        coord = 'geocentrictrueecliptic'

    if lat is not None:
        if npts > 1:
            c = []
            for lon_tmp, lat_tmp in zip(lon, lat):
                c_tmp = SkyCoord(lon_tmp, lat_tmp, unit='deg', equinox='J2000', frame=coord)
                c.append(c_tmp)
        else:
            c = SkyCoord(lon, lat, unit='deg', equinox='J2000', frame=coord)
    else:
        c = lon

    if npts == 1:
        c = c.geocentrictrueecliptic #check to see if this is the right ecliptic
        c = c.cartesian
    else:
        c2 = []
        for i in range(npts):
            c2.append(c[i].geocentrictrueecliptic)
            c2[i] = c2[i].cartesian
        c = c2
    output = _uv2pix(c, res=res)

    if npts == 1:
        return output[0]

    return output

def _uv2pix(c, res=6):
    '''Returns pixel number given unit vector pointing to center
    of pixel resolution of the cude

    Parameters
    ----------
    c : astropy.coordinates.SkyCoord or array-like (Npts, )
        sky coordinate system in cartesian coordinates

    res : int, optional
        resolution of output pixel numbers, default=6

    Returns
    -------
    pixel : int or array-like (Npts, )
        pixel numbers
    '''

    npts = np.size(c)

    res1 = res - 1
    num_pix_side = 2.0**res1
    #num_pix_face = num_pix_side**2

    face, x, y = _axisxy(c)

    i = x * num_pix_side
    i[i > 2**res1-1] = 2**res1 - 1
    i = np.array(i, dtype=np.int)

    j = y * num_pix_side
    j[j > 2**res1-1] = 2**res1 - 1
    j = np.array(j, dtype=np.int)

    fij = np.empty([npts, 3])
    fij[:, 0] = face
    fij[:, 1] = i
    fij[:, 2] = j

    pixel = _fij2pix(fij, res)

    return pixel

def _axisxy(c):
    '''Converts position into nface number (0-5) and x, y in range 0-1

    Parameters
    ----------
    c : array-like (Npts, ) of astropy.coordinates.SkyCoord objects
        sky position in cartesian coordinates
    '''

    n = np.size(c)


    if n > 1:
        c0 = np.zeros(n)
        c1 = np.zeros(n)
        c2 = np.zeros(n)
        for i in range(n):
            c0[i] = c[i].x.value
            c1[i] = c[i].y.value
            c2[i] = c[i].z.value
    else:
        c0 = np.array([c.x.value])
        c1 = np.array([c.y.value])
        c2 = np.array([c.z.value])

    abs_yx = np.ones(n)*np.inf
    abs_zx = np.ones(n)*np.inf
    abs_zy = np.ones(n)*np.inf

    g = c0 != 0
    abs_yx[g] = np.abs(c1[g]/c0[g])
    abs_zx[g] = np.abs(c2[g]/c0[g])

    g = c1 != 0
    abs_zy[g] = np.abs(c2[g]/c1[g])

    nface = 0 * np.all([abs_zx >= 1, abs_zy >= 1, c2 >= 0], axis=0) + \
            5 * np.all([abs_zx >= 1, abs_zy >= 1, c2 < 0], axis=0) + \
            1 * np.all([abs_zx < 1, abs_yx < 1, c0 >= 0], axis=0) + \
            3 * np.all([abs_zx < 1, abs_yx < 1, c0 < 0], axis=0) + \
            2 * np.all([abs_zy < 1, abs_yx >= 1, c1 >= 0], axis=0) + \
            4 * np.all([abs_zy < 1, abs_yx >= 1, c1 < 0], axis=0)

    eta = np.zeros_like(c0)
    xi = np.zeros_like(c0)

    for i in range(n):
        if nface[i] == 0:
            eta[i] = -c0[i]/c2[i]
            xi[i] = c1[i]/c2[i]
        elif nface[i] == 1:
            eta[i] = c2[i]/c0[i]
            xi[i] = c1[i]/c0[i]
        elif nface[i] == 2:
            eta[i] = c2[i]/c1[i]
            xi[i] = -c0[i]/c1[i]
        elif nface[i] == 3:
            eta[i] = -c2[i]/c0[i]
            xi[i] = c1[i]/c0[i]
        elif nface[i] == 4:
            eta[i] = -c2[i]/c1[i]
            xi[i] = -c0[i]/c1[i]
        elif nface[i] == 5:
            eta[i] = -c0[i]/c2[i]
            xi[i] = -c1[i]/c2[i]
        else:
            raise ValueError("Invalid face number")

    x, y = _incube(xi, eta)

    x = (x+1.0) / 2.0
    y = (y+1.0) / 2.0

    return nface, x, y


def _fij2pix(fij, res):
    '''This function takes an n by 3 element vector containing the
    face, column, and row number (the latter two within the face) of
    a pixel and converts it into an n-element pixel array for a given
    resolution

    Parameters
    ----------
    fij : array-like (Npts,)
        array of face, column, row numbers

    res : int
        COBE pixelization resolution for output pixel numbers

    Returns
    -------
    pixel : array-like (Npts,)
        COBE pixel numbers
    '''

    n = len(fij)

    pixel_1 = np.zeros(n, dtype=np.int)

    i = np.array(fij[:, 1], dtype=np.int)
    j = np.array(fij[:, 2], dtype=np.int)

    num_pix_face = 4**(res-1)

    pow_2 = 2**np.arange(16)

    for bit in range(res-1):
        pixel_1 = pixel_1 | ((pow_2[bit] & i) << bit)
        pixel_1 = pixel_1 | ((pow_2[bit] & j) << bit+1)

    pixel = fij[:, 0]*num_pix_face + pixel_1

    return np.array(pixel, dtype=np.int)

def _incube(alpha, beta):

    gstar = 1.37484847732
    g = -0.13161671474
    m = 0.004869491981
    w1 = -0.159596235474
    c00 = 0.141189631152
    c10 = 0.0809701286525
    c01 = -0.281528535557
    c11 = 0.15384112876
    c20 = -0.178251207466
    c02 = 0.106959469314
    d0 = 0.0759196200467
    d1 = -0.0217762490699
    aa = alpha**2
    bb = beta**2
    a4 = aa**2
    b4 = bb**2
    onmaa = 1.0 - aa
    onmbb = 1.0 - bb

    x = alpha*(gstar+aa*(1.0-gstar)+onmaa*(bb*(g+(m-g)*aa \
        +onmbb*(c00+c10*aa+c01*bb+c11*aa*bb+c20*a4+c02*b4)) \
        +aa*(w1-onmaa*(d0+d1*aa))))
    y = beta*(gstar+bb*(1.0-gstar)+onmbb*(aa*(g+(m-g)*bb \
        +onmaa*(c00+c10*bb+c01*aa+c11*bb*aa+c20*b4+c02*a4)) \
        +bb*(w1-onmbb*(d0+d1*bb))))

    return x, y

def pix2coord(pixel, res=6, coord='C'):
    '''Convert COBE quad-cube pixel number to sky coordinates (lon/lat, ra/dec, etc.)

    Parameters
    ----------
    pixel : int, array-like shape (Npts,)
        pixel number of array of pixel numbers
    res : int
        resolution of COBE pixelization. 'F'='D'=6, 'B'=9
    coord : string, optional
        coordinate system of output coordinates. Either in Healpy or Astropy coordinate string.
        Default is celestial coordinate system

    Returns
    -------
    c : astropy.coordinates.SkyCoord or array-like (Npts,)
        sky coordinates of the input pixel numbers
    '''

    if res == 'F' or res == 'D':
        res = 6
    elif res == 'B':
        res = 9

    if coord == 'C':
        coord = 'icrs'
    elif coord == 'G':
        coord = 'galactic'
    elif coord == 'E':
        coord = 'geocentrictrueecliptic'

    npts = np.size(pixel)

    scale = 2**(res-1) / 2.0

    out = _pix2fij(pixel, res)

    x = (out[:, 1] - scale + 0.5) / scale
    y = (out[:, 2] - scale + 0.5) / scale

    xi, eta = _fwdcube(x, y)

    vector = _xyaxis(out[:, 0], xi, eta)
    
    lon_lat = _uv2ll(vector)
    
    #if npts > 1:
    #    c = []
    #    for i in range(npts):
    #        print("FFF:", i)
    #        c.append(SkyCoord(lon_lat[i, 0], lon_lat[i, 1], frame='geocentrictrueecliptic',
    #                          unit='deg'))
    #        c[i] = getattr(c[i], coord)
    #else:
    #    c = SkyCoord(lon_lat[0, 0], lon_lat[0, 1], frame='geocentrictrueecliptic', unit='deg')
    #    c = getattr(c, coord)
    
    c = SkyCoord(lon_lat[:, 0], lon_lat[:, 1], frame='geocentrictrueecliptic', unit='deg')
    c = getattr(c, coord)

    return c

def _pix2fij(pixel, res=6):
    '''This function takes a n-element pixel array
    and generates an nx3 element array containing the
    corresponding face, column, and row number (the latter
    two within the face)

    Parameters
    ----------
    pixel : float or array-like (Npts,)
        pixel number array

    res : int
        COBE pixelization resolution

    Returns
    -------
    fij : array-like (Npts, 3)
        face, column, row for each pixel
    '''

    n = np.size(pixel)
    pixel = np.array(pixel)

    output = np.empty([n, 3])

    res1 = res-1

    num_pix_face = 4**res1

    face = pixel // num_pix_face
    fpix = pixel - num_pix_face*face

    output[:, 0] = face

    pow_2 = 2**np.arange(16)

    i = np.zeros(n, dtype=np.int)
    j = np.zeros(n, dtype=np.int)

    for bit in range(res):
        i = i | (pow_2[bit] * (1 & fpix))
        fpix = fpix >> 1
        j = j | (pow_2[bit] * (1 & fpix))
        fpix = fpix >> 1

    output[:, 1] = i
    output[:, 2] = j

    return output

def _fwdcube(x, y):
    '''Based on polynomial fit found using fcfit.for. Taken
    from forward_cube.for

    Parameters
    ----------
    x: float or array-like (Npts,)
        database coordinate

    y: float or array-like (Npts,)
        databse coordinate

    Returns
    -------
    xi : float or array-like (Npts,)
        tangent plane coordinate

    eta : float or array-like (Npts,)
        tangent plane coordinate
    '''

    p = np.empty(29)

    p[1] = -0.27292696
    p[2] = -0.07629969
    p[3] = -0.02819452
    p[4] = -0.22797056
    p[5] = -0.01471565
    p[6] = 0.27058160
    p[7] = 0.54852384
    p[8] = 0.48051509
    p[9] = -0.56800938
    p[10] = -0.60441560
    p[11] = -0.62930065
    p[12] = -1.74114454
    p[13] = 0.30803317
    p[14] = 1.50880086
    p[15] = 0.93412077
    p[16] = 0.25795794
    p[17] = 1.71547508
    p[18] = 0.98938102
    p[19] = -0.93678576
    p[20] = -1.41601920
    p[21] = -0.63915306
    p[22] = 0.02584375
    p[23] = -0.53022337
    p[24] = -0.83180469
    p[25] = 0.08693841
    p[26] = 0.33887446
    p[27] = 0.52032238
    p[28] = 0.14381585

    xx = x*x
    yy = y*y

    xi = x*(1.+(1.-xx)*( \
         p[1]+xx*(p[2]+xx*(p[4]+xx*(p[7]+xx*(p[11]+xx*(p[16]+xx*p[22]))))) + \
         yy*(p[3]+xx*(p[5]+xx*(p[8]+xx*(p[12]+xx*(p[17]+xx*p[23])))) + \
         yy*(p[6]+xx*(p[9]+xx*(p[13]+xx*(p[18]+xx*p[24]))) + \
         yy*(p[10]+xx*(p[14]+xx*(p[19]+xx*p[25])) + \
         yy*(p[15]+xx*(p[20]+xx*p[26]) + \
         yy*(p[21]+xx*p[27] + yy*p[28])))))))

    eta = y*(1.+(1.-yy)*( \
          p[1]+yy*(p[2]+yy*(p[4]+yy*(p[7]+yy*(p[11]+yy*(p[16]+yy*p[22]))))) + \
          xx*(p[3]+yy*(p[5]+yy*(p[8]+yy*(p[12]+yy*(p[17]+yy*p[23])))) + \
          xx*(p[6]+yy*(p[9]+yy*(p[13]+yy*(p[18]+yy*p[24]))) + \
          xx*(p[10]+yy*(p[14]+yy*(p[19]+yy*p[25])) + \
          xx*(p[15]+yy*(p[20]+yy*p[26]) + \
          xx*(p[21]+yy*p[27] + xx*p[28])))))))

    return xi, eta

def _xyaxis(nface, xi, eta):
    '''Converts face number an xi, eta into a unit vector

    Parameters
    ----------
    nface : int, array-like shape (Npts,)
        sky-cube face number
    xi : float, array-like shape (Npts,)
        tangent plane coordinate
    eta : float, array-like shape (Npts,)
        tangent plane coordinate

    Returns
    -------
    c : array-like (Npts, 3)
        unit vector
    '''

    n = np.size(nface)

    nface_0 = np.array(nface == 0, dtype=np.int)
    nface_1 = np.array(nface == 1, dtype=np.int)
    nface_2 = np.array(nface == 2, dtype=np.int)
    nface_3 = np.array(nface == 3, dtype=np.int)
    nface_4 = np.array(nface == 4, dtype=np.int)
    nface_5 = np.array(nface == 5, dtype=np.int)

    row0 = eta * (nface_5 - nface_0) + xi * (nface_4 - nface_2) + \
           (nface_1 - nface_3)
    row1 = xi * (nface_0 + nface_1 - nface_3 + nface_5) + \
           (nface_2 - nface_4)
    row2 = eta * (nface_1 + nface_2 + nface_3 + nface_4) + \
           (nface_0 - nface_5)

    norm = np.sqrt(1 + xi*xi + eta*eta)

    row0 /= norm
    row1 /= norm
    row2 /= norm

    uv = np.empty([n, 3])

    uv[:, 0] = row0
    uv[:, 1] = row1
    uv[:, 2] = row2

    return uv

def _uv2ll(vector):
    '''Convert unit vectors to longitude/latitude

    Parameters
    ----------
    vector : array-like (Npts, 3)
        unit vectors

    Returns
    -------
    lon_lat: array-like (Npts, 2)
        longitude/latitude array
    '''

    n = len(vector)

    lon_lat = np.empty([n, 2])

    lon_lat[:, 0] = np.arctan2(vector[:, 1], vector[:, 0])
    lon_lat[:, 1] = np.arcsin(vector[:, 2])

    lon_lat = np.degrees(lon_lat)

    tmp = lon_lat[:, 0]
    tmp[tmp < 0] += 360.0

    return lon_lat

def bit_table_set(ix, iy):
    '''Routine to set up the bit tables for use in the pixelization subroutines (extracted and
    generalized from existing routines)
    '''

    length = len(ix)

    for i in range(1, length+1):
        j = i-1
        k = 0
        ip = 1

        while j != 0:
            id1 = j % 2
            j //= 2
            k = ip * id1 + k
            ip *= 4
        ix[i-1] = k
        iy[i-1] = 2*k

        #if j == 0:
        #    ix[i-1] = k
        #    iy[i-1] = 2*k
        #else:
        #    id1 = j % 2
        #    j /= 2
        #    k = ip * id1 + k
        #    ip *= 4
        #    goto if statement

def edgchk(nface, ix, iy, maxval):
    '''Check for ix, iy being over the edge of a face
    Returns correct face, ix, iy in mface, jx, jy
    '''

    tempx = ix
    tempy = iy
    tempface = nface % 6

    while tempx < 0 or tempx >=  maxval or tempy < 0 or tempy >= maxval:
        if tempx < 0:
            if tempface == 0:
                mface = 4
                jy = tempx + maxval
                jx = maxval - 1 - tempy
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 1:
                mface = 4
                jx = maxval + tempx
                jy = tempy
                tempx = jx
                tempface = mface
            elif tempface == 2 or tempface == 3 or tempface == 4:
                mface = tempface - 1
                jx = maxval + tempx
                jy = tempy
                tempx = jx
                tempface = mface
            elif tempface == 5:
                mface = 4
                jx = tempy
                jy = -tempx - 1
                tempx = jx
                tempy = jy
                tempface = mface
        elif tempx >= maxval:
            if tempface == 0:
                mface = 2
                jx = tempy
                jy = 2*maxval - 1 - tempx
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 1 or tempface == 2 or tempface == 3:
                mface = tempface + 1
                jy = tempy
                jx = tempx - maxval
                tempx = jx
                tempface = mface
            elif tempface == 4:
                mface = 1
                jy = tempy
                jx = tempx - maxval
                tempx = jx
                tempface = mface
            elif tempface == 5:
                mface = 2
                jx = maxval - 1 - tempy
                jy = tempx - maxval
                tempx = jx
                tempy = jy
                tempface = mface
        elif tempy < 0:
            if tempface == 0:
                mface = 1
                jy = tempy + maxval
                jx = tempx
                tempy=  jy
                tempface = mface
            elif tempface == 1:
                mface = 5
                jy = tempy + maxval
                jx = tempx
                tempy = jy
                tempface = mface
            elif tempface == 2:
                mface = 5
                jx = tempy + maxval
                jy = maxval - 1 - tempx
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 3:
                mface = 5
                jx = maxval - 1 - tempx
                jy = -tempy - 1
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 4:
                mface = 5
                jx = -tempy - 1
                jy = tempx
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 5:
                mface = 3
                jx = maxval - 1 - tempx
                jy = -tempy - 1
                tempx = jx
                tempy = jy
                tempface = mface
        elif tempy >= maxval:
            if tempface == 0:
                mface = 3
                jx = maxval - 1 - tempx
                jy = 2*maxval - 1 - tempy
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 1:
                mface = 0
                jy = tempy - maxval
                jx = tempx
                tempy = jy
                tempface = mface
            elif tempface == 2:
                mface = 0
                jx = 2*maxval - 1 - tempy
                jy = tempx
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 3:
                mface = 0
                jx = maxval - 1 - tempx
                jy = 2*maxval - 1 - tempy
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 4:
                mface = 0
                jx = tempy - maxval
                jy = maxval - 1 - tempx
                tempx = jx
                tempy = jy
                tempface = mface
            elif tempface == 5:
                mface = 1
                jx = tempx
                jy = tempy - maxval
                tempy = jy
                tempface = mface


    mface = tempface
    jx = tempx
    jy = tempy

    return mface, jx, jy

def get_8_neighbors(pixel, res, four_neighbors=False):
    '''Generalized routine to return the numbers of all pixels adjoining the input pixel at
    the given resolution. This routine will work for resolutions up to 15.

    Parameters
    ----------
    pixel: int
        pixel number
    res: int
        resolution
    four_neighbors: bool
        return four neighbors instead of 8

    Returns
    -------
    neighbors: array, shape (8,)
        neighboring pixel numbers

    Notes
    -----
    1. Divide the pixel number up into x and y bits. These are the cartesian coordinates
       of the pixel on the face.
    2. Covert the (x, y) found from the original pixel number in the input resolution to the
       equivalent positions on a face of resolution 15. This is accomplixehd by multiplying
       by the varaince DISTANCE, which is the width of a RES pixel in resolution=15 pixels.
       Resoution=15 is the 'intermediary' format for finding neighbors.
    3. Determine the cartesian coordinates (res=15) of the neighbors by adding the appropriate
       orthogonal offset (DISTANCE) to the center pixel (x, y). EDGCHK is called to adjust the
       new coordinates and face number in case PIXEL lies along an edge.
    4. Convert the new coordinates to a resolution 15 pixel number.
    5. Convert the res=15 pixel number to a res=RES pixel number by dividing by the appropriate
       power of four (DIVISOR).

    Steps 3-5 are repeated for each of the neighbors and duplicates are deleted before being returned
    '''

    eight_neighbors = not four_neighbors

    two14 = 2**14
    two28 = 2**28

    ixtab = np.zeros(128, dtype=np.int)
    iytab = np.zeros(128, dtype=np.int)

    bit_table_set(ixtab, iytab)
    pixels_per_face = (2**(res-1)) ** 2
    face = pixel // pixels_per_face
    rpixel = pixel - face*pixels_per_face
    res_diff = 15 - res
    divisor = 4**res_diff
    distance = 2**res_diff

    ix = 0
    iy = 0
    ip = 1

    #Break pixel number down into constituent x, y coordinates
    while rpixel != 0:
        id1 = rpixel % 2
        rpixel = rpixel // 2
        ix = (id1 * ip) + ix

        id1 = rpixel % 2
        rpixel = rpixel // 2
        iy = (id1 * ip) + iy

        ip *= 2

    #Convert x, y coordinates of pixel in initial resolution to resolution of 15
    ix *= distance
    iy *= distance

    neighbors = np.zeros(4 + 4*eight_neighbors, dtype=np.int)

    #Calculate coordinates of each neighbor, check for edges, and return pixel number
    #in appropriate array element
    nface, jx, jy = edgchk(face, ix+distance, iy, two14) #Right
    jxhi = jx // 128
    jxlo = jx % 128
    jyhi = jy // 128
    jylo = jy % 128
    neighbors[0] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
    neighbors[0] /= divisor
    
    nface, jx, jy = edgchk(face, ix, iy+distance, two14) #Top
    jxhi = jx // 128
    jxlo = jx % 128
    jyhi = jy // 128
    jylo = jy % 128
    neighbors[1] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
    neighbors[1] /= divisor
    
    nface, jx, jy = edgchk(face, ix-distance, iy, two14) #Left
    jxhi = jx // 128
    jxlo = jx % 128
    jyhi = jy // 128
    jylo = jy % 128
    neighbors[2] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
    neighbors[2] /= divisor
    
    nface, jx, jy = edgchk(face, ix, iy-distance, two14) #Bottom
    jxhi = jx // 128
    jxlo = jx % 128
    jyhi = jy // 128
    jylo = jy % 128
    neighbors[3] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
    neighbors[3] /= divisor
   
    if not four_neighbors:
        nface, jx, jy = edgchk(face, ix+distance, iy+distance, two14) #Top-Right
        jxhi = jx // 128
        jxlo = jx % 128
        jyhi = jy // 128
        jylo = jy % 128
        neighbors[4] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
        neighbors[4] /= divisor
    
        nface, jx, jy = edgchk(face, ix-distance, iy+distance, two14) #Top-Left
        jxhi = jx // 128
        jxlo = jx % 128
        jyhi = jy // 128
        jylo = jy % 128
        neighbors[5] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
        neighbors[5] /= divisor
    
        nface, jx, jy = edgchk(face, ix-distance, iy-distance, two14) #Bottom-Left
        jxhi = jx // 128
        jxlo = jx % 128
        jyhi = jy // 128
        jylo = jy % 128
        neighbors[6] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
        neighbors[6] /= divisor
    
        nface, jx, jy = edgchk(face, ix+distance, iy-distance, two14) #Bottom-Right
        jxhi = jx // 128
        jxlo = jx % 128
        jyhi = jy // 128
        jylo = jy % 128
        neighbors[7] = (nface * two28) + ixtab[jxlo] + iytab[jylo] + two14 * (ixtab[jxhi] + iytab[jyhi])
        neighbors[7] /= divisor

    return np.unique(neighbors)

def get_4_neighbors(pixel, res):
    
    return get_8_neighbors(pixel, res, four_neighbors=True)

def res2npix(res):
    '''Calculates the number of pixels given the input resolution'''

    return 6 * 4**(res-1)

def npix2res(npix):
    '''Calculates the resolution given the input number of pixels'''
    
    if npix % 6 != 0:
        raise ValueError("npix does not correspond with an actual resolution")

    tmp = npix / 6

    resm1 = np.log2(tmp) / 2

    res = int(resm1 + 1)

    if 4**(res - 1) != tmp:
        raise ValueError("npix does not correspond with an actual resolution")

    return res

def sixpack(t_in, t_orient='R'):
    '''Packs an unfolded skycube into 2x3 format (no wasted space).

    Notes
    -----
    The dimensions compared to the IDL routine are reversed because IDL 
    is column-major and numpy is row-major
    '''

    #Data compression routines takes a standard unfolded skycube, i.e.
    #   00                          00
    #   00                          00
    #   11223344              44332211
    #   11223344              44332211
    #   55                          55
    #   55                          55
    #   "Left T"             "Right T"
    #
    #and compressed it into the space-saving form:
    #           445500
    #           445500
    #           332211
    #           332211
    #
    #It will work for both single "sheets" (2-d array) and spectral
    #cubes (3-D). Note that the output corresponds to a "right-T"
    #regardless of the orientation of the input

    ndim = t_in.ndim
    shape = np.shape(t_in)

    #Create degenerate third dimension if needed
    if ndim == 2:
        t_in = t_in[np.newaxis, :, :]
        depth = 1
    else:
        depth = t_in.shape[0]
    xsize = t_in.shape[1]
    ysize = t_in.shape[2]

    #Make sure this is in fact a standard unfolded cube
    if 4*xsize != 3*ysize:
        raise ValueError("This is not a standard unfolded cube!")

    #Now flip cube to right-T t_orientation if necessary.
    if t_orient == 'L':
        t_in = t_in[:, :, ::-1] #TODO: ???

    fsize = ysize/4
    box_out = t_in[:, fsize:3*fsize, fsize:4*fsize] #faces 0, 1, 2, 3
    box_out[:, fsize:2*fsize, fsize:2*fsize] = t_in[:, :fsize, 3*fsize:4*fsize] #face 5
    box_out[:, fsize:2*fsize, :fsize] = t_in[:, fsize*2*fsize, :fsize]

    if depth == 1:
        box_out.shape = (2*fsize, 3*fsize)

    return box_out

def sixunpack(box_in, badval=None):
    '''Uncompresses the packed skycybe created by sixpack
    '''

    #Create degenerate 3rd dimension if needed
    ndim = box_in.ndim
    shape = box_in.shape

    if ndim == 2:
        box_in = box_in[np.newaxis, :, :]
        depth = 1
    else:
        depth = shape[0]

    xsize = shape[1]
    ysize = shape[2]

    #Make sure that this is in fact a compressed unfolded cube
    if 3*xsize != 2*ysize:
        raise ValueError("This is not a compressed sky cube")

    #Create the output array with appropriate data type
    fsize = ysize // 3

    t_out = np.zeros([depth, 3*fsize, 4*fsize], dtype=box_in.dtype)

    #Initialize t_out to the bad pixel value if given
    if badval is not None:
        t_out[:, :, :] = badval

    #Fill in the upper left 2x3 corner of the T. Will have to zero out
    #box faces 4 and 5 later, so create a null face
    t_out[:, fsize:3*fdize, fsize:4*fsize] = box_in
    nullface = np.copy(t_out[:, :fsize, :2*fsize])

    #Fill in the 4 and 5 faces, the blank them out of the T
    t_out[:, :fsize, 3:fsize:4*fsize] = box_in[:, fsize:2*fsize, fsize:2*fsize]
    t_out[:, fsize:2*fsize, :fsize] = box_in[:, fsize:2*fsize, :fsize]
    t_out[:, 2*fsize:3*fsize, fsize:3*fsize] = nullface

    #Eliminate the degenerate dimension if needed
    if depth == 1:
        t_out.shape = (3*fsize, 4*fsize)

    return t_out

def _pix2xy(pixel, res=6, data=None, bad_pixval=0.0, face=False, sixpack=False):
    '''This function creates a raster image (sky cube or face) given a pixel list
    and data array. The data array can be either a vector or 2d-array. In the
    latter case, the data for each raster image can be stored in either
    the columns or rows. The procedure also returns the x and y raster coordinates
    of each pixel.
    '''

    pixel = np.atleast_1d(pixel)
    #pixel = np.squeeze(pixel)
    npix = len(pixel)

    switch1 = False

    if np.max(pixel) > 6*(4**(res-1)):
        raise ValueError("Maximum pixel number too large for resolution", res)

    #Determine size and "orientation" of data array
    if data is not None:
        data = np.asarray(data)

        if data.ndim == 1:
            if len(pixel) != len(data):
                raise ValueError("Pixel and data array are of incompatible size", len(pixel), len(data))
        elif data.ndim == 2:
            if len(pixel) == len(data[0]):
                switch1 = True

            if (len(pixel) != len(data)) and len(pixel) != len(data[0]):
                raise ValueError("pixel and data array are of incompatible size")
        else:
            raise ValueError("Data array must be vector or 2d array")

    if data is None:
        data = np.array([-1])
 
    if switch1:
        data = data.T

    #Call rasterization routine
    raster, x_out, y_out = _rastr(pixel, res, face=face, sixpack=sixpack, data=data,
                                  bad_pixval=bad_pixval)

    if switch1:
        data = data.T

    return x_out, y_out, raster

def _rastr(pixel, res=6, face=False, sixpack=False, data=-1, bad_pixval=0.0):
    '''Generates a raster image
    '''

    npix = len(pixel)
    ndata = np.size(data)

    if sixpack:
        #NOTE: these are packed left_t offsets
        i0 = 3
        j0 = 2
        offx = np.array([0, 0, 1, 2, 2, 1])
        offy = np.array([1, 0, 0, 0, 1, 1])
    elif face:
        #Face
        i0 = 1
        j0 = 1
        offx = np.array([0, 0, 0, 0, 0, 0])
        offy = np.array([0, 0, 0, 0, 0, 0])
    else:
        #Cube
        i0 = 4
        j0 = 3
        offx = np.array([0, 0, 1, 2, 3, 0])
        offy = np.array([2, 1, 1, 1, 1, 0])

    fij = _pix2fij(pixel, res) #get face, column, row info for pixels

    cube_side = 2**(res-1)

    len0 = i0 * cube_side

    idx = fij[:, 0].astype(np.int)

    x_out = offx[idx] * cube_side + fij[:, 1]
    x_out = (len0 - (x_out+1)).astype(np.int)
    y_out = (offy[idx] * cube_side + fij[:, 2]).astype(np.int)

    if len(data) != 1:
        thrd = ndata / npix
        raster = np.zeros([i0*cube_side, j0*cube_side, thrd])
        raster = np.squeeze(raster)

        raster += bad_pixval

        if thrd == 1:
            raster[x_out, y_out] = data
        else:
            for k in range(thrd):
                temp_arr = raster[:, :, k]
                temp_arr[x_out, y_out] = data[:, k]
                raster[:, :, k] = temp_arr
                #raster[x_out, y_out, k] = data[:, k]
    else:
        raster = None

    return raster, x_out, y_out

def _pix2dat(pixel, x_in=None, y_in=None, raster=None):
    '''This function creates a data array given either a list of
    pixels or a set of x and y raster coordinates and a raster
    image (sky cube or face). The skycube can be in either unfolded
    or six pack format. This routine is the "complement" to pix2xy.
    The program assumes a right oriented, ecliptic coordinate
    input raster.
    '''

    #Get size of input raster
    input_l, input_h = raster.shape

    if input_l == input_h:
        cube_side = input_l
    elif 3*input_l == 4*input_h:
        cube_side = input_l // 4
    elif 2*input_l == 3*input_h:
        cube_side = input_l // 3

    #Determine resolution of quad cube
    res = -1
    for bit in range(16):
        if cube_side ^ 2**bit == 0:
            res = bit + 1
            break

    if res == -1:
        raise ValueError("Improper image size")

    #Determine number of pixels / get column and row numbers if pixel entry
    if pixel.size != 0:
        num_pix = pixel.size

        if input_l == input_h:
            #face
            x_in, y_in, ras = _pix2xy(pixel, res=res, face=True)
        elif 3*input_l == 4*input_h:
            #skycube
            x_in, y_in, ras = _pix2xy(pixel, res=res)
        elif 2*input_l == 3*input_h:
            #sixpack
            x_in, y_in, ras = _pix2xy(pixel, res=res, sixpack=True)
    else:
        num_pix = x_in.size
        if x_in.size != y_in.size:
            raise ValueError("Column and Row arrays have incompatible sizes")

    #Build data array
    if len(raster.shape) == 2:
        num_ras = 1
    else:
        num_ras = raster.shape[2]

    data = np.zeros([num_pix, num_ras], dtype=raster.dtype)

    #Load data array
    if num_ras == 1:
        data = raster[x_in, y_in]
    else:
        for i in range(num_ras):
            ras1 = raster[:, :, i]
            data[:, i] = ras1[x_in, y_in]

    return data

def _uv2proj(uvec, proj, sz_proj):
    '''Converts units vectors to projection (screen) coordinates
    '''

    lon = np.arctan2(uvec[:, 1], uvec[:, 0])
    lat = np.arcsin(uvec[:, 2])

    half_l = sz_proj[1] // 2
    half_h = sz_proj[2] // 2

    if proj.upper() == 'A':
        den = np.sqrt(1.0 + np.cos(lat)*np.cos(lon / 2.0))

        proj_x = half_l - np.fix(half_l * (np.cos(lat)*np.sin(lon/2) / den)).astype(np.int)
        proj_y = half_h + np.fix(half_h * (np.sin(lat) / den)).astype(np.int)
    elif proj.upper() == 'S':
        proj_x = half_l - np.fix(half_l * lon * np.cos(lat) / np.pi).astype(np.int)
        proj_y = half_h + np.fix(half_h * lat / (np.pi/2)).astype(np.int)
    elif proj.upper() == 'M':
        pass
    elif proj.upper() == 'P':
        fac1 = np.sqrt(1 - np.sin(np.abs(lat)))
        sgn = np.sign(lat)
        fac2 = 1 - (sgn / 2)

        proj_x = fac2 * half_l - sgn * np.fix(0.5 * half_l * fac1 * np.sin(lon)).astype(np.int)
        proj_y = half_h - np.fix(half_h * fac1 * np.cos(lon)).astype(np.int)
    else:
        raise ValueError("Invalid projection string entered")

