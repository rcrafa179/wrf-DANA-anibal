#!/usr/bin/env python3
"""
batch_figuras.py -- Generador por lotes de figuras WRF para la DANA Anibal.
                    Paralelizado sobre todos los nucleos.

Reutiliza el catalogo y los diagnosticos del visor (wrfview/core.py) y sigue
el estilo del proyecto 'comparacion GFS vs ERA5': contourf en bandas, titulo
centrado de dos lineas, barra de color con "Nombre (unidades)", dpi 150 y
bbox_inches='tight'. Convencion de nombres heredada de frames_t2/ +
evolucion_t2_wrf.gif.

Salida
------
  figuras/
    frames_<var>_<dom>/<var>_frame_000.png ...     serie por instante
    evolucion_<var>_<dom>.gif                      animacion de esa serie
    resumen/<resumen>_<dom>.png                    acumulados y maximos
    cortes/corte_<var>_<dom>_<fecha>.png           cortes verticales
    metricas_evento.csv
    README.md

Paralelismo
-----------
Cada worker es un proceso con su propio Reader y sus propios Dataset abiertos:
netCDF4 no es seguro compartiendo descriptores entre procesos. Por defecto usa
todos los nucleos (--workers 0). Si la maquina empieza a paginar con d03, baja
--workers: cada proceso mantiene un wrfout abierto y los temporales de
wrf-python.

Ejemplos
--------
  python batch_figuras.py --dir ~/Z_WRF/test/em_real
  python batch_figuras.py --dir ... --dominios d03 --salto 2 --workers 8
  python batch_figuras.py --dir ... --solo-resumen --sin-cortes
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import warnings
import multiprocessing
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wrfview.core import (Inventory, Reader, Context, BY_ID, cross_section,
                          HAVE_WRFPY)
from wrfview.plot import (BaseMap, draw_field, draw_barbs, get_projection,
                          HAVE_CARTOPY, INK, INK_SOFT)

warnings.filterwarnings("ignore")

LOCAL_OFFSET = timedelta(hours=-5)          # America/Lima
DPI = 150                                   # como en comparacion GFS vs ERA5

CONJUNTOS = {
    "precip":       ["pcp_int", "pcp_acc", "snow_int", "snow_acc", "snowh",
                     "hail_acc", "graup_acc", "sr"],
    "sinoptico":    ["gph_lev", "t_lev", "rvor_lev"],
    "superficie":   ["frz_level", "t2", "wspd10", "slp"],
    "reflectividad": ["mdbz"],
}
NIVEL_SINOPTICO = 500.0

RESUMENES = [
    ("pcp_total",   "pcp_acc",   "acumulado", "Precipitacion total del evento"),
    ("snow_total",  "snow_acc",  "acumulado", "Nieve total del evento (eq. agua)"),
    ("hail_total",  "hail_acc",  "acumulado", "Granizo total del evento"),
    ("graup_total", "graup_acc", "acumulado", "Graupel total del evento"),
    ("mdbz_max",    "mdbz",      "max",       "Reflectividad maxima del evento"),
    ("wspd10_max",  "wspd10",    "max",       "Viento maximo a 10 m del evento"),
    ("frz_min",     "frz_level", "min",       "Cota de nieve mas baja del evento"),
    ("t2_min",      "t2",        "min",       "Temperatura minima a 2 m del evento"),
]
EXTREMOS = [(n, b, m) for n, b, m, _ in RESUMENES if m in ("max", "min")]


# ===========================================================================
#  Inventario ligero: viaja a los workers sin volver a escanear el disco
# ===========================================================================
class InvLite:
    """
    Reconstruye lo que Reader necesita (`frames`, `domains()`) a partir de los
    Frame ya validados. Evita que cada worker repita el escaneo de los ~283
    archivos, que en spawn (el modo por defecto en macOS) se pagaria N veces.
    """

    def __init__(self, frames, sim_start=None):
        self.frames = frames
        self.sim_start = sim_start
        self.rejected = []

    def domains(self):
        return list(self.frames)


# ===========================================================================
#  Estado de cada worker
# ===========================================================================
_W = {}


def _init_worker(frames, sim_start, dpi, modo):
    # una hebra BLAS por proceso: si no, N procesos x N hebras se estorban
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    inv = InvLite(frames, sim_start)
    # cache minima: con d03 cada proceso ya carga bastante
    _W["reader"] = Reader(inv, ds_cache=1, fld_cache=3)
    _W["dpi"] = dpi
    _W["modo"] = modo


def _spec(var_id, levels=None):
    spec = BY_ID[var_id]
    if levels is not None:
        from dataclasses import replace
        spec = replace(spec, levels=list(levels))
    return spec


# --- tarea: renderizar un frame --------------------------------------------
def _tarea_frame(t):
    dom, idx, var_id, level, levels, ruta, n = t
    try:
        reader = _W["reader"]
        spec = _spec(var_id, levels)
        data = reader.field(dom, idx, spec, level)
        uv = Context(reader, dom, idx, None).unstagger_wind10() if spec.barbs else None
        fr = reader.frame(dom, idx)
        t0 = reader.inv.frames[dom][0].time
        fig = figura_mapa(reader, dom, spec, data,
                          _titulo(spec, level), _subtitulo(reader, dom, fr, t0),
                          spec.note, barbs_uv=uv, modo=_W["modo"])
        fig.savefig(ruta, dpi=_W["dpi"], facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return (True, n, ruta, "")
    except Exception as e:
        plt.close("all")
        return (False, n, ruta, f"{type(e).__name__}: {e}")


# --- tarea: extremos parciales de un tramo de la serie ---------------------
def _tarea_extremos(t):
    """
    Devuelve los max/min parciales de un tramo y, de paso, el maximo de
    reflectividad de cada instante (sirve para situar los cortes verticales
    sin tener que recorrer los archivos otra vez).
    """
    dom, idxs, quiere_dbz = t
    reader = _W["reader"]
    parcial, picos, fallos = {}, [], {}
    for i in idxs:
        for new_id, base_id, modo in EXTREMOS:
            try:
                d = reader.field(dom, i, BY_ID[base_id], None)
            except Exception as e:
                fallos[new_id] = f"{type(e).__name__}: {e}"
                continue
            if new_id not in parcial:
                parcial[new_id] = d.copy()
            else:
                parcial[new_id] = (np.fmax(parcial[new_id], d) if modo == "max"
                                   else np.fmin(parcial[new_id], d))
            if quiere_dbz and base_id == "mdbz":
                picos.append((float(np.nanmax(d)), i))
    return parcial, picos, fallos


# --- tarea: muestreo para la escala global ---------------------------------
def _tarea_rango(t):
    dom, idx, var_id, level = t
    try:
        d = _W["reader"].field(dom, idx, BY_ID[var_id], level)
        f = d[np.isfinite(d)]
        if f.size == 0:
            return None
        return (float(np.percentile(f, 0.5)), float(np.percentile(f, 99.5)))
    except Exception:
        return None


# --- tarea: preparar un frame para el GIF ----------------------------------
def _tarea_gif_frame(t):
    """Reescala y cuantiza fuera del proceso principal; escribe un GIF suelto."""
    ruta_png, ruta_tmp, ancho = t
    try:
        from PIL import Image
        im = Image.open(ruta_png).convert("RGB")
        if ancho and im.width > ancho:
            alto = int(round(im.height * ancho / im.width))
            im = im.resize((ancho, alto), Image.LANCZOS)
        im.quantize(colors=256, method=Image.MEDIANCUT).save(ruta_tmp)
        return ruta_tmp
    except Exception:
        return None


# ===========================================================================
#  Figura estandar
# ===========================================================================
def _titulo(spec, level):
    lvl = f"  {level:.0f} hPa" if spec.pressure_level else ""
    return f"WRF — {spec.label}{lvl} — DANA Aníbal"


def _subtitulo(reader, domain, fr, t0, extra=""):
    dx = reader.dataset(fr.path).getncattr("DX") / 1000.0
    fh = (fr.time - t0).total_seconds() / 3600.0
    loc = fr.time + LOCAL_OFFSET
    s = (f"{fr.time:%Y-%m-%d %H:%M} UTC  ({loc:%d/%m %H:%M} hora de Lima)"
         f"   ·   {domain} · {dx:.0f} km · +{fh:.0f} h")
    return s + ("   ·   " + extra if extra else "")


def figura_mapa(reader, domain, spec, data, titulo, subtitulo, pie,
                barbs_uv=None, figsize=(9.4, 8.4), modo="contourf"):
    ds = reader.dataset(reader.frame(domain, 0).path)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")

    proj = get_projection(ds)
    rect = [0.07, 0.07, 0.80, 0.83]
    ax = fig.add_axes(rect, projection=proj) if proj is not None else fig.add_axes(rect)

    bm = BaseMap(ax, reader, domain)
    lat, lon = reader.coords(domain)
    tr = bm.transform if HAVE_CARTOPY else None

    mesh, _ = draw_field(ax, lon, lat, data, spec, tr, modo=modo)
    if barbs_uv is not None:
        draw_barbs(ax, lon, lat, barbs_uv[0], barbs_uv[1], tr)

    # titulo centrado de dos lineas sobre el mapa, como en frames_t2/
    ax.set_title(f"{titulo}\n{subtitulo}", fontsize=11.5, color=INK, pad=10)

    cax = fig.add_axes([0.885, 0.13, 0.022, 0.70])
    cb = fig.colorbar(mesh, cax=cax, extend=getattr(mesh, "wrfview_extend", spec.extend))
    cb.set_label(f"{spec.label} ({spec.units})", fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK_SOFT)
    cb.outline.set_linewidth(0.6)

    if pie:
        fig.text(0.07, 0.018, pie, fontsize=7.4, color=INK_SOFT)
    return fig


# ===========================================================================
#  Escala global (evita el parpadeo del GIF)
# ===========================================================================
def escala_global(pool, dom, var_id, idxs, level, muestras=12):
    spec = BY_ID[var_id]
    if spec.levels is not None:
        return None
    sel = np.unique(np.linspace(0, len(idxs) - 1,
                                min(muestras, len(idxs))).astype(int))
    tareas = [(dom, idxs[k], var_id, level) for k in sel]
    lo, hi = np.inf, -np.inf
    for r in pool.map(_tarea_rango, tareas):
        if r:
            lo, hi = min(lo, r[0]), max(hi, r[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    step = (spec.contour or {}).get("step")
    if step:
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
        return list(np.arange(lo, hi + step, step))
    return list(np.linspace(lo, hi, 16))


# ===========================================================================
#  Serie de frames + GIF, en paralelo
# ===========================================================================
def serie_frames(pool, reader, dom, var_id, outdir, salto, level, gif, fps,
                 gif_ancho, workers):
    spec = BY_ID[var_id]
    idxs = list(range(0, len(reader.inv.frames[dom]), salto))
    if spec.needs_prev and idxs and idxs[0] == 0:
        idxs = idxs[1:]                       # el primero no tiene t-1
    if not idxs:
        return None

    sufijo = f"_{level:.0f}hPa" if spec.pressure_level else ""
    nombre = f"{var_id}{sufijo}_{dom}"
    carpeta = os.path.join(outdir, f"frames_{nombre}")
    os.makedirs(carpeta, exist_ok=True)

    levels = escala_global(pool, dom, var_id, idxs, level)

    tareas = [(dom, i, var_id, level, levels,
               os.path.join(carpeta, f"{var_id}_frame_{n:03d}.png"), n)
              for n, i in enumerate(idxs)]

    hechas, fallos, t_ini = {}, [], time.time()
    for k, (ok, n, ruta, msg) in enumerate(pool.map(_tarea_frame, tareas), 1):
        if ok:
            hechas[n] = ruta
        else:
            fallos.append((n, msg))
        if k % 20 == 0 or k == len(tareas):
            el = time.time() - t_ini
            print(f"    {k:4d}/{len(tareas)} frames  "
                  f"({el:5.0f} s, {el/k:.2f} s/frame, {workers} workers)")
    for n, msg in fallos[:5]:
        print(f"    ! frame {n}: {msg}")
    if len(fallos) > 5:
        print(f"    ! ...y {len(fallos)-5} fallos mas")

    rutas = [hechas[n] for n in sorted(hechas)]
    ruta_gif = None
    if gif and len(rutas) > 1:
        destino = os.path.join(outdir, f"evolucion_{nombre}.gif")
        if _hacer_gif(pool, rutas, destino, fps, gif_ancho):
            ruta_gif = destino
            print(f"    GIF: {os.path.basename(destino)} "
                  f"({len(rutas)} frames, {os.path.getsize(destino)/1e6:.1f} MB)")
    return {"var": var_id, "dominio": dom, "frames": len(rutas),
            "carpeta": carpeta, "gif": ruta_gif, "fallos": len(fallos)}


def _hacer_gif(pool, rutas, destino, fps, ancho):
    """
    El reescalado y la cuantizacion de cada frame van al pool; el ensamblado
    final es secuencial por fuerza. Reducir el ancho es lo que de verdad
    acelera: un GIF de 169 frames a tamano completo tarda y pesa muchisimo.
    """
    try:
        from PIL import Image
    except ImportError:
        print("    ! falta Pillow; me salto el GIF")
        return False
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp(prefix="gif_")
    try:
        tareas = [(r, os.path.join(tmp, f"{i:04d}.gif"), ancho)
                  for i, r in enumerate(rutas)]
        listos = [p for p in pool.map(_tarea_gif_frame, tareas) if p]
        if len(listos) < 2:
            return False
        listos.sort()
        ims = [Image.open(p) for p in listos]
        ims[0].save(destino, save_all=True, append_images=ims[1:],
                    duration=int(1000 / max(1, fps)), loop=0, optimize=False)
        for im in ims:
            im.close()
        return True
    except Exception as e:
        print(f"    ! no pude escribir el GIF: {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
#  Resumenes del evento, en paralelo
# ===========================================================================
def resumenes_evento(pool, reader, dom, outdir, salto, workers, modo,
                     quiere_picos=True):
    carpeta = os.path.join(outdir, "resumen")
    os.makedirs(carpeta, exist_ok=True)
    frames = reader.inv.frames[dom]
    idxs = list(range(0, len(frames), salto))
    if idxs[-1] != len(frames) - 1:
        idxs.append(len(frames) - 1)

    campos = {}
    # --- acumulados: solo primero y ultimo --------------------------------
    for new_id, base_id, mo, _ in RESUMENES:
        if mo != "acumulado":
            continue
        try:
            a = reader.field(dom, idxs[0], BY_ID[base_id], None)
            b = reader.field(dom, idxs[-1], BY_ID[base_id], None)
            campos[new_id] = np.maximum(b - a, 0.0)
        except Exception as e:
            print(f"    ! {new_id}: {type(e).__name__}: {e}")

    # --- extremos: la serie se reparte en tramos entre los workers ---------
    # Cada tramo abre cada wrfout una sola vez y calcula los cuatro extremos
    # a la vez. Antes esto era una pasada por variable, ocho veces sobre 169
    # archivos de 270 MB.
    trozos = [list(c) for c in np.array_split(np.array(idxs), min(workers * 2,
                                                                  len(idxs)))]
    trozos = [c for c in trozos if c]
    t_ini, picos, fallos = time.time(), [], {}
    hecho = 0
    for parcial, pk, fl in pool.map(_tarea_extremos,
                                    [(dom, c, quiere_picos) for c in trozos]):
        for k, v in parcial.items():
            modo_k = next(m for n, b, m in EXTREMOS if n == k)
            campos[k] = v if k not in campos else (
                np.fmax(campos[k], v) if modo_k == "max" else np.fmin(campos[k], v))
        picos += pk
        fallos.update(fl)
        hecho += 1
        print(f"    pasada {hecho}/{len(trozos)} tramos  "
              f"({time.time()-t_ini:.0f} s, {workers} workers)")
    for k, v in fallos.items():
        print(f"    ! {k}: {v}")

    filas = []
    for new_id, base_id, mo, titulo in RESUMENES:
        if new_id not in campos:
            continue
        spec = BY_ID[base_id]
        data = campos[new_id]
        esp = _escala_resumen(spec, mo, data)
        t_a, t_b = frames[idxs[0]].time, frames[idxs[-1]].time
        sub = (f"{t_a:%d/%m %H:%M} → {t_b:%d/%m %H:%M} UTC "
               f"({(t_b-t_a).total_seconds()/3600:.0f} h)   ·   {dom}   ·   "
               f"{'acumulado' if mo=='acumulado' else mo+'imo'} de la serie")
        fig = figura_mapa(reader, dom, esp, data, f"WRF — {titulo}", sub,
                          spec.note, modo=modo)
        fig.savefig(os.path.join(carpeta, f"{new_id}_{dom}.png"), dpi=DPI,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)

        f = data[np.isfinite(data)]
        filas.append({
            "dominio": dom, "resumen": new_id, "variable_base": base_id,
            "modo": mo, "unidades": spec.units,
            "min": round(float(np.min(f)), 3) if f.size else "",
            "media": round(float(np.mean(f)), 3) if f.size else "",
            "p99": round(float(np.percentile(f, 99)), 3) if f.size else "",
            "max": round(float(np.max(f)), 3) if f.size else "",
            "desde": f"{t_a:%Y-%m-%d %H:%M}", "hasta": f"{t_b:%Y-%m-%d %H:%M}",
        })
        clave = "min" if mo == "min" else "max"
        print(f"    {new_id:12s} {clave}={filas[-1][clave]:>10} {spec.units}")

    picos.sort(reverse=True)
    return filas, picos


def _escala_resumen(spec, modo, data):
    """El acumulado de 4 dias no cabe en la escala de un intervalo."""
    from dataclasses import replace
    if modo != "acumulado":
        return spec
    f = data[np.isfinite(data) & (data > 0)]
    if f.size == 0:
        return spec
    top = float(np.percentile(f, 99.8))
    base = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
    lv = [x for x in base if x <= max(top, base[3])]
    return replace(spec, levels=lv if len(lv) >= 3 else base[:5])


# ===========================================================================
#  Cortes verticales
# ===========================================================================
def cortes_en_picos(reader, dom, outdir, picos, n_picos=3, transecto=None,
                    xs_var="dbz"):
    if not HAVE_WRFPY:
        print("    ! los cortes verticales necesitan wrf-python; los omito")
        return []
    if not picos:
        print("    ! no hay picos de reflectividad calculados; omito los cortes")
        return []
    carpeta = os.path.join(outdir, "cortes")
    os.makedirs(carpeta, exist_ok=True)

    lat, lon = reader.coords(dom)
    if transecto is None:
        transecto = (float(np.percentile(lat, 22)), float(np.percentile(lon, 12)),
                     float(np.percentile(lat, 72)), float(np.percentile(lon, 88)))
    p0, p1 = (transecto[0], transecto[1]), (transecto[2], transecto[3])

    hechos = []
    for val, i in picos[:n_picos]:
        fr = reader.frame(dom, i)
        try:
            ctx = Context(reader, dom, i, None)
            arr, xx, yy, label, units = cross_section(ctx, p0, p1, xs_var)
        except Exception as e:
            print(f"    ! corte {fr.time:%d/%m %H:%M}: {type(e).__name__}: {e}")
            continue

        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        cmap = "nws_ref" if xs_var == "dbz" else "YlGnBu"
        m = ax.contourf(xx, yy, np.ma.masked_invalid(arr), levels=18,
                        cmap=cmap, extend="both")
        cb = fig.colorbar(m, ax=ax, pad=0.015)
        cb.set_label(f"{label} ({units})", fontsize=9.5)
        cb.ax.tick_params(labelsize=8)

        ter = reader.terrain(dom)
        n = len(xx)
        las, los = np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)
        prof = [ter[np.unravel_index(np.argmin((lat-a)**2 + (lon-o)**2), lat.shape)]
                for a, o in zip(las, los)]
        ax.fill_between(xx, 0, prof, color="#6b5b45", zorder=5)
        ax.plot(xx, prof, color="#3d3427", lw=1.0, zorder=6)

        loc = fr.time + LOCAL_OFFSET
        ax.set_title(f"WRF — corte vertical de {label.lower()} — DANA Aníbal\n"
                     f"{fr.time:%Y-%m-%d %H:%M} UTC ({loc:%d/%m %H:%M} Lima)"
                     f"   ·   {dom}   ·   máx. en dominio {val:.0f} dBZ\n"
                     f"({p0[0]:.2f}, {p0[1]:.2f}) → ({p1[0]:.2f}, {p1[1]:.2f})",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("Distancia a lo largo del transecto (km)", fontsize=9.5)
        ax.set_ylabel("Altura (m snm)", fontsize=9.5)
        ax.set_ylim(0, min(16000, float(np.nanmax(yy))))
        ax.grid(True, lw=0.3, ls=":", color="#c8c8c8")
        ax.tick_params(labelsize=8, colors=INK_SOFT)
        fig.tight_layout()
        ruta = os.path.join(carpeta,
                            f"corte_{xs_var}_{dom}_{fr.time:%Y%m%d_%H%M}.png")
        fig.savefig(ruta, dpi=DPI, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        hechos.append(ruta)
        print(f"    corte {fr.time:%d/%m %H:%M} (máx {val:.0f} dBZ)")
    return hechos


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generador por lotes de figuras WRF (DANA Aníbal)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", "-d", default=".", help="carpeta con los wrfout_*")
    ap.add_argument("--salida", "-o", default="figuras")
    ap.add_argument("--dominios", default="d01,d02,d03")
    ap.add_argument("--conjuntos", default="precip,sinoptico,superficie,reflectividad",
                    help=f"subconjunto de: {','.join(CONJUNTOS)}")
    ap.add_argument("--vars", default=None,
                    help="ids de variable explicitos (anula --conjuntos)")
    ap.add_argument("--salto", type=int, default=1,
                    help="1 de cada N instantes (d03 cada 30 min: --salto 2 = horario)")
    ap.add_argument("--workers", "-w", type=int, default=0,
                    help="procesos en paralelo; 0 = todos los nucleos")
    ap.add_argument("--nivel", type=float, default=NIVEL_SINOPTICO)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--gif-ancho", type=int, default=1000,
                    help="ancho en px de los frames del GIF; 0 = sin reescalar")
    ap.add_argument("--pcolormesh", action="store_true",
                    help="malla sin suavizar en vez de contourf (mas fiel en d03)")
    ap.add_argument("--sin-gif", action="store_true")
    ap.add_argument("--sin-resumen", action="store_true")
    ap.add_argument("--sin-cortes", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    ap.add_argument("--transecto", default=None, help="lat0,lon0,lat1,lon1")
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    modo = "pcolormesh" if args.pcolormesh else "contourf"

    inv = Inventory(args.dir)
    reader = Reader(inv, ds_cache=2, fld_cache=6)
    os.makedirs(args.salida, exist_ok=True)

    doms = [d for d in args.dominios.split(",") if d in inv.domains()]
    if not doms:
        print(f"Ninguno de {args.dominios} existe. Hay: {inv.domains()}")
        return 1

    if args.vars:
        variables = [v.strip() for v in args.vars.split(",") if v.strip() in BY_ID]
    else:
        variables = []
        for c in args.conjuntos.split(","):
            variables += CONJUNTOS.get(c.strip(), [])
        variables = list(dict.fromkeys(variables))

    transecto = tuple(float(x) for x in args.transecto.split(",")) \
        if args.transecto else None

    print(f"\nDominios : {doms}")
    print(f"Variables: {variables}")
    print(f"Salto    : 1 de cada {args.salto}")
    print(f"Workers  : {workers} de {os.cpu_count()} núcleos")
    print(f"Marca    : {modo}")
    print(f"Salida   : {os.path.abspath(args.salida)}\n")

    t_total = time.time()
    metricas, hechos = [], []

    # 'spawn' a proposito, no 'fork': heredar por fork descriptores netCDF/HDF5
    # ya abiertos puede corromper lecturas. Ademas es lo que usa macOS por
    # defecto, asi el comportamiento es el mismo en Linux y en Mac.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker,
            initargs=(inv.frames, inv.sim_start, args.dpi, modo)) as pool:

        for dom in doms:
            print(f"\n{'='*70}\n{dom} — {len(inv.frames[dom])} instantes\n{'='*70}")
            picos = []

            if not args.solo_resumen:
                for vid in variables:
                    spec = BY_ID[vid]
                    lvl = args.nivel if spec.pressure_level else None
                    etq = f"{vid}{f' {lvl:.0f}hPa' if lvl else ''}"
                    print(f"\n  [{etq}] {spec.label}")
                    r = serie_frames(pool, reader, dom, vid, args.salida,
                                     args.salto, lvl, not args.sin_gif,
                                     args.fps, args.gif_ancho, workers)
                    if r:
                        hechos.append(r)

            if not args.sin_resumen:
                print(f"\n  [resumen] acumulados y máximos del evento")
                filas, picos = resumenes_evento(
                    pool, reader, dom, args.salida, args.salto, workers, modo,
                    quiere_picos=not args.sin_cortes)
                metricas += filas

            if not args.sin_cortes:
                print(f"\n  [cortes] verticales en los picos de reflectividad")
                cortes_en_picos(reader, dom, args.salida, picos,
                                transecto=transecto)

    if metricas:
        ruta_csv = os.path.join(args.salida, "metricas_evento.csv")
        with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(metricas[0].keys()))
            w.writeheader()
            w.writerows(metricas)
        print(f"\nMétricas: {ruta_csv}")

    _escribir_readme(args.salida, inv, variables, hechos, metricas, args, workers)
    reader.close()
    print(f"\nListo en {(time.time()-t_total)/60:.1f} min → "
          f"{os.path.abspath(args.salida)}")
    return 0


def _escribir_readme(outdir, inv, variables, hechos, metricas, args, workers):
    L = ["# Figuras — DANA Aníbal", "",
         f"Generadas desde `{os.path.abspath(args.dir)}` con {workers} procesos.", "",
         f"Inicio de simulación: **{inv.sim_start}**", "",
         "## Datos de entrada", "",
         "| Dominio | Instantes | Desde | Hasta | Paso |", "|---|---|---|---|---|"]
    for d in inv.domains():
        f = inv.frames[d]
        paso = ((f[1].time - f[0].time).total_seconds() / 60) if len(f) > 1 else 0
        L.append(f"| {d} | {len(f)} | {f[0].time:%Y-%m-%d %H:%M} | "
                 f"{f[-1].time:%Y-%m-%d %H:%M} | {paso:.0f} min |")
    if inv.rejected:
        L += ["", f"Se descartaron {len(inv.rejected)} archivos "
                  "(truncados o de otra corrida)."]
    if hechos:
        L += ["", "## Series y animaciones", "",
              "| Variable | Dominio | Frames | Carpeta | GIF |", "|---|---|---|---|---|"]
        for h in hechos:
            g = os.path.basename(h["gif"]) if h["gif"] else "—"
            L.append(f"| {h['var']} | {h['dominio']} | {h['frames']} | "
                     f"`{os.path.basename(h['carpeta'])}/` | `{g}` |")
    if metricas:
        L += ["", "## Resúmenes del evento", "",
              "En `resumen/`. Cifras completas en `metricas_evento.csv`.", "",
              "| Resumen | Dominio | Valor extremo | Unidades |", "|---|---|---|---|"]
        for m in metricas:
            clave = "min" if m["modo"] == "min" else "max"
            L.append(f"| {m['resumen']} | {m['dominio']} | {clave} {m[clave]} "
                     f"| {m['unidades']} |")
    L += ["", "## Notas", "",
          "- Estilo heredado del proyecto de comparación GFS vs ERA5: `contourf` "
          "en bandas, título centrado de dos líneas, barra de color con "
          "«Nombre (unidades)», dpi 150 y `bbox_inches='tight'`.",
          "- Las escalas de color son fijas a lo largo de cada serie, para que "
          "las animaciones no parpadeen.",
          "- `pcp_int` y `snow_int` son diferencias con el instante anterior, "
          "así que la serie empieza en el segundo instante.",
          "- Los acumulados del evento son último menos primero, no la suma de "
          "intervalos: los contadores de WRF ya son monótonos.",
          "- Los cortes verticales se sitúan solos en los instantes de mayor "
          "reflectividad, reaprovechando el barrido de los resúmenes.", ""]
    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
