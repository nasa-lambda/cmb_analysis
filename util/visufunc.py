import matplotlib
import numpy as np
from astropy.coordinates import CartesianRepresentation
import healpy.projaxes as PA
import healpy.rotator as R

from . import pixfunc

def vec2pix(res, x, y, z):
    c = CartesianRepresentation(x, y, z)
    return pixfunc._uv2pix(c, res=res)

def mollview(
    map=None,
    fig=None,
    #rot=None,
    coord=None,
    #unit="",
    #xsize=800,
    title="Mollweide view",
    min=None,
    max=None,
    #flip="astro",
    #remove_dip=False,
    remove_mono=False,
    gal_cut=0,
    #format="%g",
    #format2="%g",
    #cbar=True,
    #cmap=None,
    #badcolor="gray",
    #bgcolor="white",
    #notext=False,
    #norm=None,
    #hold=False,
    #margins=None,
    #sub=None,
    #nlocs=2,
    return_projected_map=False,
):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure()

    #TODO: Update with better values
    rect = [0, 0, 1, 1]

    if coord is not None:
        if type(coord) is str:
            coord = ['E', coord]
        elif type(coord) is list:
            if coord[0] != 'E':
                raise ValueError("First coordinate must be 'E'")
    else:
        coord = 'E'

    ax1 = QcMollweideAxes(fig, rect, coord=coord)
    fig.add_axes(ax1)

    if remove_mono:
        pixfunc.remove_monopole(map, gal_cut=gal_cut)

    img = ax1.projmap(map, vmin=min, vmax=max, coord=coord)

    if title is not None:
        ax1.set_title(title)

    if return_projected_map:
        return img

class QcGnomonicAxes(PA.GnomonicAxes):
    def projmap(self, map, **kwds):
        res = pixfunc.npix2res(pixfunc.get_map_size(map))
        f = lambda x, y, z: vec2pix(res, x, y, z)
        xsize = kwds.pop("xsize", 200)
        ysize = kwds.pop("ysize", None)
        reso = kwds.pop("reso", 1.5)
        return super(QcGnomonicAxes, self).projmap(
            map, f, xsize=xsize, ysize=ysize, reso=reso, **kwds
        )

class QcMollweideAxes(PA.MollweideAxes):
    def projmap(self, map, **kwds):
        res = pixfunc.npix2res(pixfunc.get_map_size(map))
        f = lambda x, y, z: vec2pix(res, x, y, z)
        return super(QcMollweideAxes, self).projmap(map, f, **kwds)

    def projquiver(self, *args, **kwds):
        """projquiver is a wrapper around :func:`matplotlib.Axes.quiver` to take
        into account the spherical projection.

        You can call this function as::

           projquiver([X, Y], U, V, [C], **kw)
           projquiver([theta, phi], mag, ang, [C], **kw)

        Parameters
        ----------

        Notes
        -----
        Other keywords are transmitted to :func:`matplotlib.Axes.quiver`

        See Also
        --------
        projplot, projscatter, projtext
        """

        '''
        Qmap = args[0][0]
        Umap = args[0][1]

        npix_in = len(Qmap)
        nside_in = pixfunc.npix2nside(npix_in)

        Qmap = pixfunc.ud_grade(Qmap, nside)
        Umap = pixfunc.ud_grade(Umap, nside)

        mag = np.sqrt(Qmap**2 + Umap**2)
        ang = np.arctan2(Umap, Qmap) / 2.0

        max_mag = np.max(mag)
        mag /= max_mag

        npix = H.nside2npix(nside)
        pix = np.arange(npix)

        theta, phi = pixfunc.pix2ang(nside, pix)
        '''

        #Work on input signature
        c = None
        if len(args) < 2:
            raise ValueError("Not enough arguments given")
        if len(args) == 2:
            mag, ang = np.asarray(args[0]), np.asarray(args[1])
            raise ValueError("Not calculating theta, phi yet")
        elif len(args) == 4:
            theta, phi = np.asarray(args[0]), np.asarray(args[1])
            mag, ang = np.asarray(args[2]), np.asarray(args[3])
        elif len(args) == 5:
            theta, phi = np.asarray(args[0]), np.asarray(args[1])
            mag, ang = np.asarray(args[2]), np.asarray(args[3])
            c = np.asarray(args[4])
        else:
            raise TypeError("Wrong number of arguments given")

        save_input_data = hasattr(self.figure, "zoomtool")
        if save_input_data:
            input_data = (theta, phi, mag, ang, args, kwds.copy())

        rot = kwds.pop("rot", None)
        if rot is not None:
            rot = np.array(np.atleast_1d(rot), copy=1)
            rot.resize(3)
            rot[1] = rot[1] - 90.

        coord = self.proj.mkcoord(kwds.pop("coord", None))[::-1]
        lonlat = kwds.pop("lonlat", False)
    
        vec = R.dir2vec(theta, phi, lonlat=lonlat)
        vec = (R.Rotator(rot=rot, coord=coord, eulertype="Y")).I(vec)
        x, y = self.proj.vec2xy(vec, direct=kwds.pop("direct", False))
   
        lat = np.pi/2 - theta
        lon = np.pi - phi

        ang_off = -np.pi/2*np.sin(lat)*np.sin(lon)
    
        #ang_off = np.zeros_like(ang)

        u = 1e-10*mag*np.cos(ang + ang_off)
        v = 1e-10*mag*np.sin(ang + ang_off)

        #x = np.linspace(-2.0, 2.0, num=10)
        #y = np.zeros_like(x)
        #u = 1e-5
        #v = 1e-5

        #s = self.quiver(x, y, u, v, c, *args, headwidth=0, width=0.001, pivot='mid', **kwds)
        #s = self.quiver(x, y, u, v, c, *args, headwidth=0, width=0.01, pivot='mid', **kwds)
        if c is not None:
            s = self.quiver(x, y, u, v, c, headwidth=0, width=0.001, pivot='mid', **kwds)
        else:
            s = self.quiver(x, y, u, v, headwidth=0, width=0.001, pivot='mid', **kwds)

        '''
        theta = 0.0
        phi = 0.0
        
        s = self.projtext(theta, phi, 'AA')
        vec = R.dir2vec(theta, phi, lonlat=lonlat)
        vec = (R.Rotator(rot=rot, coord=None, eulertype="Y")).I(vec)
        x, y = self.proj.vec2xy(vec, direct=kwds.pop("direct", False))
        print("TEST:", x, y)
        '''

        if save_input_data:
            if not hasattr(self, "_quiver_data"):
                self._quiver_data = []
            self._quiver_data.append((s, input_data))
        return s
    

class QcCartesianAxes(PA.CartesianAxes):
    def projmap(self, map, nest=False, **kwds):
        res = pixfunc.npix2res(pixfunc.get_map_size(map))
        f = lambda x, y, z: vec2pix(res, x, y, z)
        return super(QcCartesianAxes, self).projmap(map, f, **kwds)

class QcOrthographicAxes(PA.OrthographicAxes):
    def projmap(self, map, nest=False, **kwds):
        res = pixfunc.npix2res(pixfunc.get_map_size(map))
        f = lambda x, y, z: vec2pix(res, x, y, z)
        return super(QcOrthographicAxes, self).projmap(map, f, **kwds)

class QcAzimuthalAxes(PA.AzimuthalAxes):
    def projmap(self, map, nest=False, **kwds):
        res = pixfunc.npix2res(pixfunc.get_map_size(map))
        f = lambda x, y, z: vec2pix(res, x, y, z)
        xsize = kwds.pop("xsize", 800)
        ysize = kwds.pop("ysize", None)
        reso = kwds.pop("reso", 1.5)
        lamb = kwds.pop("lamb", True)
        return super(QcAzimuthalAxes, self).projmap(
            map, f, xsize=xsize, ysize=ysize, reso=reso, lamb=lamb, **kwds
        )

