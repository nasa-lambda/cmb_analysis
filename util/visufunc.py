import matplotlib
import numpy as np
from astropy.coordinates import CartesianRepresentation
import healpy.projaxes as PA 

from . import pixfunc


def vec2pix(res, x, y, z):
    c = CartesianRepresentation(x, y, z)
    return pixfunc._uv2pix(c, res=res)

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

def mollview(
    map=None,
    fig=None,
    rot=None,
    coord=None,
    unit="",
    xsize=800,
    title="Mollweide view",
    nest=False,
    min=None,
    max=None,
    flip="astro",
    remove_dip=False,
    remove_mono=False,
    gal_cut=0,
    format="%g",
    format2="%g",
    cbar=True,
    cmap=None,
    badcolor="gray",
    bgcolor="white",
    notext=False,
    norm=None,
    hold=False,
    margins=None,
    sub=None,
    nlocs=2,
    return_projected_map=False,
):
    """Plot a healpix map (given as an array) in Mollweide projection.
    
    Parameters
    ----------
    map : float, array-like or None
      An array containing the map, supports masked maps, see the `ma` function.
      If None, will display a blank map, useful for overplotting.
    fig : int or None, optional
      The figure number to use. Default: create a new figure
    rot : scalar or sequence, optional
      Describe the rotation to apply.
      In the form (lon, lat, psi) (unit: degrees) : the point at
      longitude *lon* and latitude *lat* will be at the center. An additional rotation
      of angle *psi* around this direction is applied.
    coord : sequence of character, optional
      Either one of 'G', 'E' or 'C' to describe the coordinate
      system of the map, or a sequence of 2 of these to rotate
      the map from the first to the second coordinate system.
    unit : str, optional
      A text describing the unit of the data. Default: ''
    xsize : int, optional
      The size of the image. Default: 800
    title : str, optional
      The title of the plot. Default: 'Mollweide view'
    nest : bool, optional
      If True, ordering scheme is NESTED. Default: False (RING)
    min : float, optional
      The minimum range value
    max : float, optional
      The maximum range value
    flip : {'astro', 'geo'}, optional
      Defines the convention of projection : 'astro' (default, east towards left, west towards right)
      or 'geo' (east towards right, west towards left)
    remove_dip : bool, optional
      If :const:`True`, remove the dipole+monopole
    remove_mono : bool, optional
      If :const:`True`, remove the monopole
    gal_cut : float, scalar, optional
      Symmetric galactic cut for the dipole/monopole fit.
      Removes points in latitude range [-gal_cut, +gal_cut]
    format : str, optional
      The format of the scale label. Default: '%g'
    format2 : str, optional
      Format of the pixel value under mouse. Default: '%g'
    cbar : bool, optional
      Display the colorbar. Default: True
    notext : bool, optional
      If True, no text is printed around the map
    norm : {'hist', 'log', None}
      Color normalization, hist= histogram equalized color mapping,
      log= logarithmic color mapping, default: None (linear color mapping)
    cmap : a color map
       The colormap to use (see matplotlib.cm)
    badcolor : str
      Color to use to plot bad values
    bgcolor : str
      Color to use for background
    hold : bool, optional
      If True, replace the current Axes by a MollweideAxes.
      use this if you want to have multiple maps on the same
      figure. Default: False
    sub : int, scalar or sequence, optional
      Use only a zone of the current figure (same syntax as subplot).
      Default: None
    margins : None or sequence, optional
      Either None, or a sequence (left,bottom,right,top)
      giving the margins on left,bottom,right and top
      of the axes. Values are relative to figure (0-1).
      Default: None
    return_projected_map : bool
      if True returns the projected map in a 2d numpy array

    See Also
    --------
    gnomview, cartview, orthview, azeqview
    """
    # Create the figure
    import pylab

    # Ensure that the resolution is valid
    res = pixfunc.get_res(map)

    if not (hold or sub):
        f = pylab.figure(fig, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
    elif hold:
        f = pylab.gcf()
        left, bottom, right, top = np.array(f.gca().get_position()).ravel()
        extent = (left, bottom, right - left, top - bottom)
        f.delaxes(f.gca())
    else:  # using subplot syntax
        f = pylab.gcf()
        if hasattr(sub, "__len__"):
            nrows, ncols, idx = sub
        else:
            nrows, ncols, idx = sub // 100, (sub % 100) // 10, (sub % 10)
        if idx < 1 or idx > ncols * nrows:
            raise ValueError("Wrong values for sub: %d, %d, %d" % (nrows, ncols, idx))
        c, r = (idx - 1) % ncols, (idx - 1) // ncols
        if not margins:
            margins = (0.01, 0.0, 0.0, 0.02)
        extent = (
            c * 1.0 / ncols + margins[0],
            1.0 - (r + 1) * 1.0 / nrows + margins[1],
            1.0 / ncols - margins[2] - margins[0],
            1.0 / nrows - margins[3] - margins[1],
        )
        extent = (
            extent[0] + margins[0],
            extent[1] + margins[1],
            extent[2] - margins[2] - margins[0],
            extent[3] - margins[3] - margins[1],
        )
        # extent = (c*1./ncols, 1.-(r+1)*1./nrows,1./ncols,1./nrows)
    # f=pylab.figure(fig,figsize=(8.5,5.4))

    # Starting to draw : turn interactive off
    wasinteractive = pylab.isinteractive()
    pylab.ioff()
    try:
        if map is None:
            map = np.zeros(12) + np.inf
            cbar = False
        #map = pixelfunc.ma_to_array(map)
        ax = QcMollweideAxes(
            f, extent, coord=coord, rot=rot, format=format2, flipconv=flip
        )
        f.add_axes(ax)
        #if remove_dip:
        #    map = pixelfunc.remove_dipole(
        #        map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True
        #    )
        #elif remove_mono:
        #    map = pixelfunc.remove_monopole(
        #        map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True
        #    )
        img = ax.projmap(
            map,
            xsize=xsize,
            coord=coord,
            vmin=min,
            vmax=max,
            cmap=cmap,
            badcolor=badcolor,
            bgcolor=bgcolor,
            norm=norm,
        )
        if cbar:
            im = ax.get_images()[0]
            b = im.norm.inverse(np.linspace(0, 1, im.cmap.N + 1))
            v = np.linspace(im.norm.vmin, im.norm.vmax, im.cmap.N)
            if matplotlib.__version__ >= "0.91.0":
                cb = f.colorbar(
                    im,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.5,
                    aspect=25,
                    ticks=PA.BoundaryLocator(nlocs, norm),
                    pad=0.05,
                    fraction=0.1,
                    boundaries=b,
                    values=v,
                    format=format,
                )
            else:
                # for older matplotlib versions, no ax kwarg
                cb = f.colorbar(
                    im,
                    orientation="horizontal",
                    shrink=0.5,
                    aspect=25,
                    ticks=PA.BoundaryLocator(nlocs, norm),
                    pad=0.05,
                    fraction=0.1,
                    boundaries=b,
                    values=v,
                    format=format,
                )
            cb.solids.set_rasterized(True)
        ax.set_title(title)
        if not notext:
            ax.text(
                0.86,
                0.05,
                ax.proj.coordsysstr,
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes,
            )
        if cbar:
            cb.ax.text(
                0.5,
                -1.0,
                unit,
                fontsize=14,
                transform=cb.ax.transAxes,
                ha="center",
                va="center",
            )
        f.sca(ax)
    finally:
        pylab.draw()
        if wasinteractive:
            pylab.ion()
            # pylab.show()
    if return_projected_map:
        return img

