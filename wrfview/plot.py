"""
plot.py -- Capa grafica: proyeccion, mapas base, paletas y superposiciones.

Criterio de color
-----------------
  * Las variables que ya existian en el proyecto de comparacion GFS vs ERA5
    conservan la paleta de alli: YlGnBu precipitacion, RdYlBu_r temperatura,
    viridis viento, hot_r CAPE, Blues CIN, cividis SLP, BuGn humedad.
  * Las nuevas siguen el mismo criterio: magnitud -> rampa secuencial
    monotona en luminosidad; polaridad -> rampa divergente con neutro en el
    centro (nivel de 0 C centrado en 4000 m, vorticidad en 0).
  * Unica excepcion deliberada: la escala de reflectividad, que sigue la
    convencion de radar porque se lee por color, no por rampa.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

warnings.filterwarnings("ignore")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:                                    # pragma: no cover
    ccrs = cfeature = None
    HAVE_CARTOPY = False

try:
    import wrf as wrfpy
    HAVE_WRFPY = True
except Exception:                                    # pragma: no cover
    wrfpy = None
    HAVE_WRFPY = False


# ---------------------------------------------------------------------------
#  Paleta de reflectividad (convencion de radar)
# ---------------------------------------------------------------------------
_NWS_REF = [
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6", "#4b0082",
]
if "nws_ref" not in mpl.colormaps:
    mpl.colormaps.register(ListedColormap(_NWS_REF, name="nws_ref"))


# --- ciudades del aviso SENAMHI --------------------------------------------
CITIES = [
    ("Lima",         -12.05, -77.04),
    ("Huanuco",       -9.93, -76.24),
    ("Cerro de Pasco", -10.68, -76.26),
    ("Huancayo",     -12.07, -75.21),
    ("Huancavelica", -12.79, -74.97),
    ("Pisco",        -13.71, -76.20),
    ("Ayacucho",     -13.16, -74.22),
    ("Abancay",      -13.64, -72.88),
    ("Cusco",        -13.53, -71.97),
    ("Nazca",        -14.83, -74.94),
    ("Puno",         -15.84, -70.03),
    ("Arequipa",     -16.41, -71.54),
    ("Moquegua",     -17.19, -70.94),
    ("Tacna",        -18.01, -70.25),
    ("Juliaca",      -15.50, -70.13),
]

INK = "#1a1a1a"
INK_SOFT = "#5c5c5c"

# --- disponibilidad de Natural Earth ---------------------------------------
_NE_SCALE = "10m"
_NE_STATE: Optional[bool] = None


def _naturalearth_available() -> bool:
    """
    Comprueba (una sola vez) si cartopy puede resolver los shapefiles de
    Natural Earth. Los materializa a proposito: si solo se anade la capa, el
    fallo aparece recien al dibujar y no se puede capturar.

    Prueba 10m y, si no, 50m: cartopy los descarga la primera vez que se usan
    y necesita internet en esa primera ejecucion.
    """
    global _NE_STATE, _NE_SCALE
    if _NE_STATE is not None:
        return _NE_STATE
    for scale in ("10m", "50m", "110m"):
        try:
            feat = cfeature.NaturalEarthFeature("physical", "coastline", scale)
            next(iter(feat.geometries()))
            _NE_SCALE, _NE_STATE = scale, True
            return True
        except Exception:
            continue
    _NE_STATE = False
    return False


def make_norm(spec, data: np.ndarray):
    """Norma discreta a partir de los niveles del catalogo (o de los datos)."""
    levels = spec.levels
    if levels is None:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            levels = [0, 1]
        else:
            lo, hi = np.percentile(finite, [1, 99])
            if spec.contour and spec.contour.get("step"):
                step = spec.contour["step"]
                lo = np.floor(lo / step) * step
                hi = np.ceil(hi / step) * step
                levels = list(np.arange(lo, hi + step, step))
            else:
                levels = list(np.linspace(lo, hi, 16))
        if len(levels) < 2:
            levels = [0, 1]
    # Extension efectiva. En un campo continuo (viento, PBL, agua precipitable)
    # hay que extender tambien por abajo: si no, contourf deja sin rellenar
    # todo lo que cae bajo el primer nivel y salen agujeros blancos donde en
    # realidad hay calma. Solo los campos donde el cero es ausencia
    # (mask_below) se dejan deliberadamente en blanco.
    ext = spec.extend
    if not spec.mask_below:
        ext = {"max": "both", "neither": "min"}.get(ext, ext)

    # BoundaryNorm exige ncolors == n_bins + extensiones. Reescalamos la
    # paleta para que siempre cuadre, sea continua o discreta (nws_ref).
    n_ext = {"neither": 0, "min": 1, "max": 1, "both": 2}[ext]
    needed = (len(levels) - 1) + n_ext
    cmap = plt.get_cmap(spec.cmap)
    if cmap.N != needed:
        cmap = cmap.resampled(needed)
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend=ext)
    return cmap, norm, levels, ext


def get_projection(ds):
    """CRS de cartopy leido de los atributos globales del wrfout."""
    if HAVE_CARTOPY and HAVE_WRFPY:
        try:
            return wrfpy.get_cartopy(wrfin=ds)
        except Exception:
            pass
    if HAVE_CARTOPY:
        return ccrs.PlateCarree()
    return None


def make_axes(fig, rect, ds):
    proj = get_projection(ds)
    if proj is not None:
        ax = fig.add_axes(rect, projection=proj)
    else:
        ax = fig.add_axes(rect)
    return ax


class BaseMap:
    """
    Dibuja el fondo del mapa una sola vez por dominio y lo reutiliza.

    Si cartopy no puede descargar Natural Earth (primera ejecucion sin
    internet), cae a dibujar la linea de costa a partir de LANDMASK del
    propio wrfout, que siempre esta disponible.
    """

    def __init__(self, ax, reader, domain: str, show_cities=True,
                 show_terrain=True):
        self.ax, self.reader, self.domain = ax, reader, domain
        self.lat, self.lon = reader.coords(domain)
        self.artists = []
        self.geo_source = "—"
        self._draw(show_cities, show_terrain)

    # -- transformacion de datos ------------------------------------------
    @property
    def transform(self):
        return ccrs.PlateCarree() if HAVE_CARTOPY else self.ax.transData

    def _draw(self, show_cities, show_terrain):
        ax = self.ax
        pc = dict(transform=ccrs.PlateCarree()) if HAVE_CARTOPY else {}

        # --- costas y fronteras -------------------------------------------
        # Cartopy descarga Natural Earth al DIBUJAR, no al anadir la capa, asi
        # que hay que forzar la descarga aqui para poder capturar el fallo.
        drew = False
        if HAVE_CARTOPY and _naturalearth_available():
            try:
                ax.add_feature(cfeature.COASTLINE.with_scale(_NE_SCALE),
                               linewidth=0.7, edgecolor=INK, zorder=6)
                ax.add_feature(cfeature.BORDERS.with_scale(_NE_SCALE),
                               linewidth=0.5, edgecolor=INK_SOFT, zorder=6)
                try:
                    ax.add_feature(cfeature.STATES.with_scale(_NE_SCALE),
                                   linewidth=0.3, edgecolor="#9a9a9a", zorder=6)
                except Exception:
                    pass
                drew, self.geo_source = True, f"Natural Earth {_NE_SCALE} (cartopy)"
            except Exception:
                drew = False
        if not drew:
            lm = self.reader.landmask(self.domain)
            ax.contour(self.lon, self.lat, lm, levels=[0.5], colors=[INK],
                       linewidths=0.9, zorder=6, **pc)
            self.geo_source = "LANDMASK del wrfout (cartopy sin datos locales)"

        # --- relieve: 2000 y 4000 m, las cotas del aviso ------------------
        if show_terrain:
            ter = self.reader.terrain(self.domain)
            if np.nanmax(ter) > 1500:
                ax.contour(self.lon, self.lat, ter, levels=[2000, 4000],
                           colors=["#8a6d3b", "#5c4425"], linewidths=[0.5, 0.7],
                           alpha=0.55, zorder=5, **pc)

        # --- ciudades ------------------------------------------------------
        if show_cities:
            la0, la1 = float(self.lat.min()), float(self.lat.max())
            lo0, lo1 = float(self.lon.min()), float(self.lon.max())
            mla, mlo = 0.03 * (la1 - la0), 0.03 * (lo1 - lo0)
            for name, la, lo in CITIES:
                if la0 + mla < la < la1 - mla and lo0 + mlo < lo < lo1 - mlo:
                    ax.plot(lo, la, "o", ms=3.2, mfc="white", mec=INK,
                            mew=0.8, zorder=8, **pc)
                    ax.text(lo + 0.10, la + 0.06, name, fontsize=6.6,
                            color=INK, zorder=8,
                            path_effects=_halo(), **pc)

        # --- reticula ------------------------------------------------------
        if HAVE_CARTOPY:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#c8c8c8",
                              alpha=0.7, linestyle=":")
            gl.top_labels = gl.right_labels = False
            gl.x_inline = gl.y_inline = False      # etiquetas fuera del mapa
            # en Lambert cartopy gira las etiquetas siguiendo la reticula
            # y en un dominio alto se encavalgan; horizontales se leen
            gl.rotate_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK_SOFT}
            # Sin esto cartopy autoescala al ultimo artista y recorta el
            # dominio: el minimo de la DANA se perdia por el borde sur.
            ax.set_extent([float(self.lon.min()), float(self.lon.max()),
                           float(self.lat.min()), float(self.lat.max())],
                          crs=ccrs.PlateCarree())
        else:
            ax.grid(True, lw=0.3, color="#c8c8c8", ls=":")
            ax.set_xlim(self.lon.min(), self.lon.max())
            ax.set_ylim(self.lat.min(), self.lat.max())
        for s in ax.spines.values():
            s.set_edgecolor("#b0b0b0")
            s.set_linewidth(0.8)


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.8, foreground="white")]


def draw_field(ax, lon, lat, data, spec, transform=None, modo="pcolormesh"):
    """
    Relleno principal + contorno opcional. Devuelve (mappable, artistas).

    modo='pcolormesh' respeta la malla tal cual (mejor para el visor y para
    la reflectividad en d03); modo='contourf' suaviza en bandas, que es el
    estilo del proyecto de comparacion GFS vs ERA5.
    """
    cmap, norm, levels, ext = make_norm(spec, data)
    kw = {"transform": transform} if transform is not None else {}
    arts = []

    plotted = np.ma.masked_invalid(data)
    if spec.mask_below and spec.levels is not None:
        # solo donde el cero significa ausencia (precipitacion < 0.1 mm).
        # En campos continuos como el viento dejaria agujeros blancos.
        plotted = np.ma.masked_less(plotted, levels[0])

    if modo == "contourf":
        mesh = ax.contourf(lon, lat, plotted, levels=levels, cmap=cmap,
                           norm=norm, extend=ext, zorder=2, **kw)
    else:
        mesh = ax.pcolormesh(lon, lat, plotted, cmap=cmap, norm=norm,
                             shading="auto", zorder=2, **kw)
    # la barra de color debe usar la MISMA extension que el relleno
    mesh.wrfview_extend = ext
    arts.append(mesh)

    if spec.contour:
        clev = spec.contour.get("levels")
        if clev is None and spec.contour.get("step"):
            step = spec.contour["step"]
            f = data[np.isfinite(data)]
            if f.size:
                clev = np.arange(np.floor(f.min() / step) * step,
                                 np.ceil(f.max() / step) * step + step, step)
        if clev is not None and len(np.atleast_1d(clev)):
            cs = ax.contour(lon, lat, data, levels=clev,
                            colors=spec.contour.get("color", INK),
                            linewidths=spec.contour.get("lw", 0.9),
                            zorder=7, **kw)
            arts.append(cs)
            if spec.contour.get("fmt"):
                lbl = ax.clabel(cs, inline=True, fontsize=6.2,
                                fmt=spec.contour["fmt"])
                arts.append(lbl)
    return mesh, arts


def draw_barbs(ax, lon, lat, u, v, transform=None, step=None):
    """Barbas de viento raleadas para que no saturen el mapa."""
    ny, nx = u.shape
    if step is None:
        step = max(1, int(round(max(ny, nx) / 26.0)))
    kw = {"transform": transform} if transform is not None else {}
    sl = (slice(None, None, step), slice(None, None, step))
    return ax.barbs(np.asarray(lon)[sl], np.asarray(lat)[sl],
                    u[sl] * 1.94384, v[sl] * 1.94384,   # m/s -> nudos
                    length=4.6, linewidth=0.5, color=INK, zorder=9, **kw)
