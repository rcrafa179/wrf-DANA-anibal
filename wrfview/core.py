"""
core.py -- Inventario de archivos wrfout, lectura con cache y catalogo de
variables diagnosticas para el visor de la DANA Anibal.

Todo el calculo cientifico vive aqui. La capa grafica no sabe nada de WRF.
"""
from __future__ import annotations

import os
import re
import glob
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, List, Dict

import numpy as np
import netCDF4 as nc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import wrf as wrfpy
    HAVE_WRFPY = True
except Exception:                                    # pragma: no cover
    wrfpy = None
    HAVE_WRFPY = False

FNAME_RE = re.compile(
    r"^wrfout_(d\d{2})_(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):(\d{2})$"
)

G = 9.81


# ===========================================================================
#  Inventario
# ===========================================================================
@dataclass
class Frame:
    """Un archivo wrfout = un instante de un dominio."""
    path: str
    domain: str
    time: datetime
    size: int
    ok: bool = True
    reason: str = ""


class Inventory:
    """
    Escanea un directorio de salida de WRF y construye la lista de instantes
    utilizables por dominio.

    Descarta, avisando, dos cosas que aparecen en corridas reales:
      - archivos truncados (disco lleno): pesan menos que el tamano modal
        del dominio, y al abrirlos netCDF falla o faltan variables;
      - archivos sobrantes de otra corrida: su atributo SIMULATION_START_DATE
        no coincide con el de la corrida mayoritaria.
    """

    def __init__(self, directory: str, verbose: bool = True):
        self.directory = directory
        self.verbose = verbose
        self.frames: Dict[str, List[Frame]] = {}
        self.rejected: List[Frame] = []
        self.sim_start: Optional[str] = None
        self._scan()

    # -- escaneo ------------------------------------------------------------
    def _scan(self):
        paths = sorted(glob.glob(os.path.join(self.directory, "wrfout_d*")))
        if not paths:
            raise SystemExit(
                f"No encontre archivos wrfout_* en {self.directory}\n"
                f"Usa:  python wrfview.py --dir /ruta/a/em_real"
            )

        cand: List[Frame] = []
        for p in paths:
            m = FNAME_RE.match(os.path.basename(p))
            if not m:
                continue
            dom = m.group(1)
            t = datetime(*(int(m.group(i)) for i in range(2, 8)))
            cand.append(Frame(p, dom, t, os.path.getsize(p)))

        # 1) filtro por tamano modal, por dominio -> detecta truncados
        by_dom: Dict[str, List[Frame]] = {}
        for fr in cand:
            by_dom.setdefault(fr.domain, []).append(fr)

        for dom, frs in by_dom.items():
            modal = Counter(f.size for f in frs).most_common(1)[0][0]
            for f in frs:
                if f.size < modal:
                    f.ok, f.reason = False, (
                        f"truncado ({f.size/1e6:.0f} MB de {modal/1e6:.0f} MB)"
                    )

        # 2) fecha de inicio de simulacion mayoritaria -> descarta otra corrida
        starts = Counter()
        probe = [f for frs in by_dom.values() for f in frs if f.ok]
        for f in probe:
            s = self._read_sim_start(f)
            if s:
                starts[s] += 1
        if starts:
            self.sim_start = starts.most_common(1)[0][0]
            for f in probe:
                s = self._read_sim_start(f)
                if s is None:
                    f.ok, f.reason = False, "no se pudo abrir"
                elif s != self.sim_start:
                    f.ok, f.reason = False, f"otra corrida (inicio {s})"

        for dom, frs in sorted(by_dom.items()):
            good = sorted([f for f in frs if f.ok], key=lambda f: f.time)
            if good:
                self.frames[dom] = good
            self.rejected += [f for f in frs if not f.ok]

        if self.verbose:
            self.report()

    _startcache: Dict[str, Optional[str]] = {}

    def _read_sim_start(self, f: Frame) -> Optional[str]:
        if f.path in self._startcache:
            return self._startcache[f.path]
        val = None
        try:
            with nc.Dataset(f.path) as ds:
                for k in ("SIMULATION_START_DATE", "START_DATE"):
                    if k in ds.ncattrs():
                        val = ds.getncattr(k)
                        break
        except Exception:
            val = None
        self._startcache[f.path] = val
        return val

    # -- informe ------------------------------------------------------------
    def report(self):
        print("=" * 72)
        print(f"Inventario de {self.directory}")
        if self.sim_start:
            print(f"Inicio de simulacion: {self.sim_start}")
        print("=" * 72)
        for dom, frs in self.frames.items():
            step = ""
            if len(frs) > 1:
                dt = (frs[1].time - frs[0].time).total_seconds() / 60
                step = f", cada {dt:.0f} min"
            print(f"  {dom}: {len(frs):4d} instantes utilizables{step}")
            print(f"        {frs[0].time:%Y-%m-%d %H:%M}  ->  {frs[-1].time:%Y-%m-%d %H:%M} UTC")
        if self.rejected:
            print(f"\n  {len(self.rejected)} archivo(s) descartado(s):")
            byreason = Counter(f.reason.split(" (")[0] for f in self.rejected)
            for r, n in byreason.most_common():
                ej = next(f for f in self.rejected if f.reason.startswith(r))
                print(f"     {n:4d} x {r:22s} p.ej. {os.path.basename(ej.path)}")
        print("=" * 72)

    def domains(self) -> List[str]:
        return list(self.frames.keys())


# ===========================================================================
#  Lectura con cache
# ===========================================================================
class Reader:
    """Mantiene abiertos unos pocos Datasets y cachea los campos calculados."""

    def __init__(self, inventory: Inventory, ds_cache: int = 4, fld_cache: int = 60):
        self.inv = inventory
        self._ds: OrderedDict[str, nc.Dataset] = OrderedDict()
        self._fld: OrderedDict[tuple, np.ndarray] = OrderedDict()
        # Suelo de 2: diagnosticos como pcp_int abren el instante anterior
        # mientras el actual sigue en uso (Context.prev). Con cache de 1 la
        # expulsion cierra el Dataset que se esta leyendo y netCDF revienta
        # con "NetCDF: Not a valid ID".
        self._ds_max, self._fld_max = max(2, ds_cache), fld_cache

    def dataset(self, path: str) -> nc.Dataset:
        if path in self._ds:
            self._ds.move_to_end(path)
            return self._ds[path]
        ds = nc.Dataset(path)
        self._ds[path] = ds
        while len(self._ds) > self._ds_max:
            _, old = self._ds.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return ds

    def frame(self, domain: str, idx: int) -> Frame:
        return self.inv.frames[domain][idx]

    def field(self, domain: str, idx: int, spec: "VarSpec",
              level: Optional[float] = None) -> np.ndarray:
        key = (domain, idx, spec.id, level)
        if key in self._fld:
            self._fld.move_to_end(key)
            return self._fld[key]
        ctx = Context(self, domain, idx, level)
        val = np.asarray(spec.compute(ctx), dtype=float)
        self._fld[key] = val
        while len(self._fld) > self._fld_max:
            self._fld.popitem(last=False)
        return val

    def coords(self, domain: str):
        """(lat2d, lon2d) del dominio; identicas en todos los instantes."""
        key = ("__coords__", domain, None, None)
        if key not in self._fld:
            ds = self.dataset(self.frame(domain, 0).path)
            self._fld[key] = (ds.variables["XLAT"][0], ds.variables["XLONG"][0])
        return self._fld[key]

    def terrain(self, domain: str) -> np.ndarray:
        key = ("__ter__", domain, None, None)
        if key not in self._fld:
            ds = self.dataset(self.frame(domain, 0).path)
            self._fld[key] = np.asarray(ds.variables["HGT"][0])
        return self._fld[key]

    def landmask(self, domain: str) -> np.ndarray:
        key = ("__lm__", domain, None, None)
        if key not in self._fld:
            ds = self.dataset(self.frame(domain, 0).path)
            self._fld[key] = np.asarray(ds.variables["LANDMASK"][0])
        return self._fld[key]

    def close(self):
        for ds in self._ds.values():
            try:
                ds.close()
            except Exception:
                pass
        self._ds.clear()


class Context:
    """Lo que recibe cada funcion de diagnostico."""

    def __init__(self, reader: Reader, domain: str, idx: int,
                 level: Optional[float]):
        self.reader, self.domain, self.idx, self.level = reader, domain, idx, level
        self.ds = reader.dataset(reader.frame(domain, idx).path)
        self.time = reader.frame(domain, idx).time

    # --- acceso crudo ------------------------------------------------------
    def raw(self, name: str) -> np.ndarray:
        return np.asarray(self.ds.variables[name][0])

    def has(self, name: str) -> bool:
        return name in self.ds.variables

    @property
    def prev(self) -> Optional["Context"]:
        if self.idx == 0:
            return None
        return Context(self.reader, self.domain, self.idx - 1, self.level)

    # --- diagnosticos via wrf-python, con respaldo propio ------------------
    def gv(self, name: str, **kw):
        """wrf.getvar con mensaje claro si wrf-python no esta instalado."""
        if not HAVE_WRFPY:
            raise RuntimeError(
                f"'{name}' necesita wrf-python.\n"
                f"  conda install -c conda-forge wrf-python"
            )
        return np.asarray(wrfpy.getvar(self.ds, name, **kw))

    def pressure(self) -> np.ndarray:
        """Presion total en hPa (3D), sin depender de wrf-python."""
        return (self.raw("P") + self.raw("PB")) / 100.0

    def height(self) -> np.ndarray:
        """Altura geopotencial en m sobre niveles de masa (3D)."""
        ph = (self.raw("PH") + self.raw("PHB")) / G
        return 0.5 * (ph[:-1] + ph[1:])

    def tempc(self) -> np.ndarray:
        """Temperatura en C (3D)."""
        th = self.raw("T") + 300.0
        p = (self.raw("P") + self.raw("PB"))
        return th * (p / 1.0e5) ** 0.2854 - 273.15

    def to_level(self, field3d: np.ndarray, plev: float) -> np.ndarray:
        """Interpola un campo 3D al nivel de presion plev (hPa)."""
        if HAVE_WRFPY:
            return np.asarray(wrfpy.interplevel(field3d, self.pressure(), plev))
        return _interp_lin(field3d, self.pressure(), plev, decreasing=True)

    def unstagger_wind10(self):
        """(u10, v10) rotados a coordenadas terrestres (norte verdadero)."""
        u, v = self.raw("U10"), self.raw("V10")
        if self.has("SINALPHA") and self.has("COSALPHA"):
            sa, ca = self.raw("SINALPHA"), self.raw("COSALPHA")
            return u * ca - v * sa, v * ca + u * sa
        return u, v


def _interp_lin(field, vcoord, target, decreasing=True):
    """Interpolacion lineal columna a columna (respaldo sin wrf-python)."""
    nz, ny, nx = field.shape
    out = np.full((ny, nx), np.nan)
    sign = -1.0 if decreasing else 1.0
    v = sign * vcoord
    tgt = sign * target
    for k in range(nz - 1):
        lo, hi = v[k], v[k + 1]
        m = (lo <= tgt) & (tgt < hi) & np.isnan(out)
        if m.any():
            w = (tgt - lo[m]) / np.where((hi - lo)[m] == 0, np.nan, (hi - lo)[m])
            out[m] = field[k][m] + w * (field[k + 1][m] - field[k][m])
    return out


# ===========================================================================
#  Catalogo de variables
# ===========================================================================
@dataclass
class VarSpec:
    id: str
    label: str
    group: str
    units: str
    compute: Callable[[Context], np.ndarray]
    cmap: str = "viridis"
    levels: Optional[list] = None
    extend: str = "max"
    diverging: bool = False
    pressure_level: bool = False        # usa el selector de nivel
    needs_prev: bool = False            # requiere el instante anterior
    barbs: bool = False                 # superpone viento
    # Deja en blanco lo que cae bajo el primer nivel. Correcto donde el cero
    # significa "no paso nada" (lluvia, nieve, granizo, reflectividad), y
    # ERRONEO en campos continuos: el viento en calma no es un agujero.
    mask_below: bool = False
    contour: Optional[dict] = None      # {'levels':[...], 'fmt':'%d', 'color':...}
    note: str = ""


# --- niveles discretos reutilizables ---------------------------------------
PCP_LEV = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]
SNOW_LEV = [0.1, 0.5, 1, 2, 4, 6, 10, 15, 20, 30, 50, 80]
HAIL_LEV = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15]
T_LEV = list(np.arange(-24, 25, 2.0))
DBZ_LEV = list(np.arange(5, 76, 5.0))


# --- funciones de diagnostico ----------------------------------------------
def _pcp_total(c: Context) -> np.ndarray:
    tot = c.raw("RAINNC") + c.raw("RAINC")
    if c.has("RAINSH"):
        tot = tot + c.raw("RAINSH")
    # cubetas (si el namelist activo bucket_mm)
    if c.has("I_RAINNC"):
        tot = tot + 100.0 * (c.raw("I_RAINNC") + c.raw("I_RAINC"))
    return tot


def _pcp_interval(c: Context) -> np.ndarray:
    p = c.prev
    if p is None:
        return np.zeros_like(_pcp_total(c))
    return np.maximum(_pcp_total(c) - _pcp_total(p), 0.0)


def _snow_interval(c: Context) -> np.ndarray:
    p = c.prev
    if p is None:
        return np.zeros_like(c.raw("SNOWNC"))
    return np.maximum(c.raw("SNOWNC") - p.raw("SNOWNC"), 0.0)


def _freezing_level(c: Context) -> np.ndarray:
    """Altura (m snm) de la isoterma de 0 C. Bajo el suelo -> terreno."""
    z, tc = c.height(), c.tempc()
    if HAVE_WRFPY:
        frz = np.asarray(wrfpy.interplevel(z, tc, 0.0))
    else:
        frz = _interp_lin(z, tc, 0.0, decreasing=True)
    # donde toda la columna esta bajo 0 C, el nivel esta en superficie
    ter = c.raw("HGT")
    allcold = np.nanmax(tc, axis=0) < 0.0
    frz = np.where(allcold, ter, frz)
    return frz


def _rel_vort(c: Context) -> np.ndarray:
    """Vorticidad relativa (1e-5 s-1) en el nivel de presion pedido."""
    avo = c.gv("avo")                       # absoluta, 1e-5 s-1
    f = c.raw("F") * 1e5                    # Coriolis a las mismas unidades
    return c.to_level(avo - f[None, :, :], c.level)


def _wspd_level(c: Context) -> np.ndarray:
    u, v = c.gv("ua", units="m s-1"), c.gv("va", units="m s-1")
    return np.hypot(c.to_level(u, c.level), c.to_level(v, c.level))


def _w_max(c: Context) -> np.ndarray:
    w = c.raw("W")
    w = 0.5 * (w[:-1] + w[1:])
    return np.nanmax(w, axis=0)


def _pw(c: Context) -> np.ndarray:
    """Agua precipitable (mm) integrando QVAPOR en la masa de cada capa."""
    if HAVE_WRFPY:
        return c.gv("pw")
    q = c.raw("QVAPOR")
    ph = (c.raw("PH") + c.raw("PHB")) / G
    dz = np.diff(ph, axis=0)
    p = (c.raw("P") + c.raw("PB"))
    tk = (c.raw("T") + 300.0) * (p / 1.0e5) ** 0.2854
    rho = p / (287.0 * tk * (1 + 0.61 * q))
    return np.sum(q * rho * dz, axis=0)


def _rh2(c: Context) -> np.ndarray:
    if HAVE_WRFPY:
        return c.gv("rh2")
    t = c.raw("T2") - 273.15
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = c.raw("Q2") * c.raw("PSFC") / 100.0 / (0.622 + 0.378 * c.raw("Q2"))
    return np.clip(100.0 * e / es, 0, 100)


def build_catalog() -> List[VarSpec]:
    V = []

    # ---------- 1. Precipitacion e hidrometeoros ---------------------------
    g = "1. Precipitacion / hidrometeoros"
    V += [
        VarSpec("pcp_acc", "Precipitacion acumulada", g, "mm", _pcp_total,
                cmap="YlGnBu", levels=PCP_LEV, mask_below=True,
                note="RAINC + RAINNC + RAINSH desde el inicio de la simulacion"),
        VarSpec("pcp_int", "Precipitacion del intervalo", g, "mm", _pcp_interval,
                cmap="YlGnBu", levels=[0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30],
                needs_prev=True, mask_below=True,
                note="Diferencia con el instante anterior del mismo dominio"),
        VarSpec("snow_acc", "Nieve acumulada (eq. agua)", g, "mm",
                lambda c: c.raw("SNOWNC"), cmap="BuPu", levels=SNOW_LEV,
                mask_below=True),
        VarSpec("snow_int", "Nieve del intervalo (eq. agua)", g, "mm",
                _snow_interval, cmap="BuPu",
                levels=[0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12], needs_prev=True,
                mask_below=True),
        VarSpec("snowh", "Espesor fisico de nieve", g, "cm",
                lambda c: c.raw("SNOWH") * 100.0, cmap="BuPu",
                levels=[0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100], mask_below=True),
        VarSpec("hail_acc", "Granizo acumulado", g, "mm",
                lambda c: c.raw("HAILNC"), cmap="PuRd", levels=HAIL_LEV,
                mask_below=True),
        VarSpec("graup_acc", "Graupel acumulado", g, "mm",
                lambda c: c.raw("GRAUPELNC"), cmap="PuRd", levels=HAIL_LEV,
                mask_below=True),
        VarSpec("sr", "Fraccion de precipitacion congelada", g, "0-1",
                lambda c: c.raw("SR"), cmap="Blues",
                levels=list(np.arange(0.05, 1.01, 0.05)), extend="neither",
                note="SR = 1 -> toda la precipitacion cae como nieve/hielo"),
    ]

    # ---------- 2. Superficie ---------------------------------------------
    g = "2. Superficie"
    V += [
        VarSpec("t2", "Temperatura a 2 m", g, "C",
                lambda c: c.raw("T2") - 273.15, cmap="RdYlBu_r", levels=T_LEV,
                extend="both", diverging=True,
                contour={"levels": [0.0], "color": "#1a1a1a", "lw": 1.6, "fmt": "%d C"},
                note="Isoterma de 0 C resaltada en negro"),
        VarSpec("td2", "Punto de rocio a 2 m", g, "C",
                lambda c: c.gv("td2"), cmap="BrBG", levels=list(np.arange(-20, 25, 2.5)),
                extend="both", diverging=True),
        VarSpec("rh2", "Humedad relativa a 2 m", g, "%", _rh2,
                cmap="BuGn", levels=list(np.arange(10, 101, 5)), extend="neither"),
        VarSpec("slp", "Presion a nivel del mar", g, "hPa",
                lambda c: c.gv("slp"), cmap="cividis",
                levels=list(np.arange(994, 1035, 2.0)), extend="both", diverging=True,
                contour={"levels": list(np.arange(980, 1041, 2.0)),
                         "color": "#1a1a1a", "lw": 0.8, "fmt": "%d"},
                barbs=True),
        VarSpec("wspd10", "Viento a 10 m", g, "m/s",
                lambda c: np.hypot(*c.unstagger_wind10()), cmap="viridis",
                levels=[1, 2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30], barbs=True,
                note="U10/V10 rotados a norte verdadero con SINALPHA/COSALPHA"),
        VarSpec("pblh", "Altura de la capa limite", g, "m",
                lambda c: c.raw("PBLH"), cmap="YlOrBr",
                levels=list(np.arange(100, 3001, 200))),
        VarSpec("tsk", "Temperatura de piel", g, "C",
                lambda c: c.raw("TSK") - 273.15, cmap="RdYlBu_r", levels=T_LEV,
                extend="both", diverging=True),
    ]

    # ---------- 3. Sinoptico: la DANA -------------------------------------
    g = "3. Sinoptico (la DANA)"
    V += [
        VarSpec("gph_lev", "Geopotencial", g, "dam",
                lambda c: c.to_level(c.height(), c.level) / 10.0,
                cmap="viridis", levels=None, extend="both",
                pressure_level=True,
                contour={"levels": None, "step": 6.0, "color": "#1a1a1a",
                         "lw": 0.9, "fmt": "%d"},
                note="El nucleo frio aislado se ve como un minimo cerrado en 500 hPa"),
        VarSpec("t_lev", "Temperatura", g, "C",
                lambda c: c.to_level(c.tempc(), c.level),
                cmap="RdYlBu_r", levels=list(np.arange(-40, 21, 2.0)),
                extend="both", diverging=True, pressure_level=True),
        VarSpec("wspd_lev", "Viento (jet)", g, "m/s", _wspd_level,
                cmap="viridis", levels=[10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
                pressure_level=True),
        VarSpec("rvor_lev", "Vorticidad relativa", g, "1e-5 s-1", _rel_vort,
                cmap="RdBu_r", levels=list(np.arange(-30, 31, 4.0)),
                extend="both", diverging=True, pressure_level=True,
                note="Negativa = ciclonica en el hemisferio sur"),
        # Divergente centrada en 4000 m a proposito: lo que importa no es la
        # altura en si sino si queda por debajo o por encima del umbral de
        # nieve del aviso. Azul = isoterma de 0 C mas baja que 4000 m.
        VarSpec("frz_level", "Altura del nivel de 0 C", g, "m", _freezing_level,
                cmap="RdBu_r", levels=list(np.arange(2400, 5601, 200)),
                extend="both", diverging=True,
                contour={"levels": [4000.0], "color": "#111111", "lw": 2.0,
                         "fmt": "%d m"},
                note="Centrada en 4000 m, el umbral de nieve del aviso SENAMHI: "
                     "azul = nivel de 0 C por debajo de esa cota"),
    ]

    # ---------- 4. Conveccion y estructura vertical ------------------------
    g = "4. Conveccion / vertical"
    V += [
        VarSpec("mdbz", "Reflectividad maxima simulada", g, "dBZ",
                lambda c: c.gv("mdbz"), cmap="nws_ref", levels=DBZ_LEV, mask_below=True,
                note="Derivada de QRAIN/QSNOW/QGRAUP: REFL_10CM no se escribio"),
        VarSpec("w_max", "Velocidad vertical maxima", g, "m/s", _w_max,
                cmap="Reds", levels=[0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8],
                mask_below=True),
        VarSpec("pw", "Agua precipitable", g, "mm", _pw,
                cmap="YlGnBu", levels=list(np.arange(4, 61, 4.0))),
        VarSpec("cape", "CAPE de la parcela mas inestable", g, "J/kg",
                lambda c: c.gv("cape_2d")[0], cmap="hot_r",
                levels=[50, 100, 250, 500, 750, 1000, 1500, 2000, 3000],
                mask_below=True),
        VarSpec("cin", "CIN", g, "J/kg",
                lambda c: c.gv("cape_2d")[1], cmap="Blues",
                levels=[10, 25, 50, 100, 150, 200, 300, 500], mask_below=True),
        VarSpec("ctt", "Temperatura del tope nuboso", g, "C",
                lambda c: c.gv("ctt"), cmap="Greys",
                levels=list(np.arange(-70, 21, 5.0)), extend="both"),
        VarSpec("cf_low", "Nubosidad baja", g, "fraccion",
                lambda c: c.gv("cloudfrac")[0], cmap="Greys",
                levels=list(np.arange(0.05, 1.01, 0.05)), extend="neither"),
        VarSpec("cf_mid", "Nubosidad media", g, "fraccion",
                lambda c: c.gv("cloudfrac")[1], cmap="Greys",
                levels=list(np.arange(0.05, 1.01, 0.05)), extend="neither"),
        VarSpec("cf_high", "Nubosidad alta", g, "fraccion",
                lambda c: c.gv("cloudfrac")[2], cmap="Greys",
                levels=list(np.arange(0.05, 1.01, 0.05)), extend="neither"),
    ]
    return V


CATALOG = build_catalog()
BY_ID = {v.id: v for v in CATALOG}
GROUPS = list(dict.fromkeys(v.group for v in CATALOG))


def group_vars(group: str) -> List[VarSpec]:
    return [v for v in CATALOG if v.group == group]


# ===========================================================================
#  Corte vertical
# ===========================================================================
def cross_section(ctx: Context, p0, p1, varname: str = "dbz", nz: int = 60):
    """
    Corte vertical entre dos puntos (lat, lon).
    Devuelve (matriz 2D, eje x en km, eje y en m, etiqueta, unidades).
    """
    if not HAVE_WRFPY:
        raise RuntimeError("El corte vertical necesita wrf-python.")
    ds = ctx.ds
    z = wrfpy.getvar(ds, "z")
    start = wrfpy.CoordPair(lat=p0[0], lon=p0[1])
    end = wrfpy.CoordPair(lat=p1[0], lon=p1[1])

    fields = {
        "dbz":    ("dbz", "Reflectividad", "dBZ"),
        "tc":     ("tc", "Temperatura", "C"),
        "rh":     ("rh", "Humedad relativa", "%"),
        "wa":     ("wa", "Velocidad vertical", "m/s"),
        "theta_e": ("theta_e", "Temperatura potencial equivalente", "K"),
    }
    if varname == "qhydro":
        q = (wrfpy.getvar(ds, "QSNOW") + wrfpy.getvar(ds, "QGRAUP")
             + wrfpy.getvar(ds, "QRAIN") + wrfpy.getvar(ds, "QCLOUD")
             + wrfpy.getvar(ds, "QICE")) * 1000.0
        var, label, units = q, "Hidrometeoros totales", "g/kg"
    else:
        key, label, units = fields.get(varname, fields["dbz"])
        var = wrfpy.getvar(ds, key)

    xs = wrfpy.vertcross(var, z, wrfin=ds, start_point=start, end_point=end,
                         latlon=True, meta=True, autolevels=nz)
    yy = np.array([float(str(v).split("_")[0]) for v in xs.coords["vertical"].values])
    dist = _haversine(p0, p1)
    xx = np.linspace(0, dist, xs.shape[-1])
    return np.asarray(xs), xx, yy, label, units


def _haversine(a, b) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(h))
