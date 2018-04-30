# pylint: disable=E1101, C0103, R0912, R0913, R0914, R0915, W0212

#Copyright 2016 United States Government as represented by the Administrator
#of the National Aeronautics and Space Administration. All Rights Reserved.

'''This module provides functions to convert between sky coordinates and
COBE pixel numbers. This is adapted from the IDL code written in support
of COBE data analysis.
'''

import numpy as np
from astropy.coordinates import SkyCoord

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

def pix2coord(pixel, res, coord='C'):
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

    if npts > 1:
        c = []
        for i in range(npts):
            c.append(SkyCoord(lon_lat[i, 0], lon_lat[i, 1], frame='geocentrictrueecliptic',
                              unit='deg'))
            c[i] = getattr(c[i], coord)
    else:
        c = SkyCoord(lon_lat[0, 0], lon_lat[0, 1], frame='geocentrictrueecliptic', unit='deg')
        c = getattr(c, coord)

    return c

def _pix2fij(pixel, res):
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
            j /= 2
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

def get_8_neighbors(pixel, res):
    '''Generalized routine to return the numbers of all pixels adjoining the input pixel at
    the given resolution

    Parameters
    ----------
    pixel: int
        pixel number
    res: int
        resolution

    Returns
    -------
    neighbors: array, shape (8,)
        neighboring pixel numbers
    number_of_neighbors : int
        number of adjoining pixels (4-8)

    Notes
    -----
    1. All eight adjoining pixels are found by manipulation of the cartesian coordinates of the
       input pixel in the plane of the face on which it lies. Set UPX_4_NEIGHBORS opening comments
       for a detailed description of this process
    2. Once all eight neighbors are found, the neighbors array is sorted and duplicate pixel numbers
       are deleted
    3. The number of unique neighbors and the array of pixel numbers are returned
    '''

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

    neighbors = np.zeros(8, dtype=np.int)

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

    neighbors = np.unique(neighbors)

    return neighbors
