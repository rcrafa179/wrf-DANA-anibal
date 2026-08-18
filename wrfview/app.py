"""
app.py -- Visor interactivo de salidas WRF.

Controles
---------
  Dominio / grupo / variable   paneles de radio a la izquierda
  Nivel de presion             panel inferior izquierdo (solo variables 3D)
  Tiempo                       deslizador inferior, o teclas <- ->
  Shift + <- ->                salta 6 instantes
  Arriba / Abajo               variable anterior / siguiente del grupo
  Espacio                      anima / pausa
  g                            guarda la figura actual en PNG
  x                            modo corte vertical: hace clic en dos puntos
  c                            cambia la variable del corte vertical
  Mover el raton               lee el valor bajo el cursor
"""
from __future__ import annotations

import os
import warnings
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from .core import (Inventory, Reader, Context, CATALOG, GROUPS, group_vars,
                   cross_section, HAVE_WRFPY)
from .plot import (BaseMap, draw_field, draw_barbs, make_axes, HAVE_CARTOPY,
                   INK, INK_SOFT)

warnings.filterwarnings("ignore")

PRESSURE_LEVELS = [850, 700, 500, 300, 250, 200]
XS_VARS = ["dbz", "qhydro", "tc", "rh", "wa", "theta_e"]
LOCAL_OFFSET = timedelta(hours=-5)          # America/Lima


class WRFViewer:
    def __init__(self, directory: str, outdir: str = "figuras",
                 domain: str | None = None, var: str = "pcp_int"):
        self.inv = Inventory(directory)
        self.reader = Reader(self.inv)
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        self.domain = domain if domain in self.inv.domains() else self.inv.domains()[0]
        self.spec = next((v for v in CATALOG if v.id == var), CATALOG[0])
        self.group = self.spec.group
        self.level = 500.0
        self.tidx = 0
        self.playing = False
        self.xs_mode = False
        self.xs_pts = []
        self.xs_var = "dbz"

        self._field_artists = []
        self._barb_artist = None
        self._cbar = None
        self.basemap = None

        self._build_figure()
        self.redraw(full=True)

    # ==================================================================
    #  Construccion de la interfaz
    # ==================================================================
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15.2, 9.0))
        self.fig.canvas.manager.set_window_title("Visor WRF - DANA Anibal")
        self.fig.patch.set_facecolor("white")

        # --- paneles de control -------------------------------------------
        self.ax_dom = self.fig.add_axes([0.015, 0.855, 0.16, 0.105])
        self.ax_grp = self.fig.add_axes([0.015, 0.665, 0.16, 0.165])
        self.ax_var = self.fig.add_axes([0.015, 0.275, 0.16, 0.365])
        self.ax_lev = self.fig.add_axes([0.015, 0.155, 0.16, 0.100])
        for a, t in ((self.ax_dom, "Dominio"), (self.ax_grp, "Grupo"),
                     (self.ax_var, "Variable"), (self.ax_lev, "Nivel (hPa)")):
            a.set_title(t, fontsize=8, color=INK, loc="left", pad=3)
            a.set_facecolor("#fafafa")

        self.rb_dom = RadioButtons(self.ax_dom, self.inv.domains(),
                                   active=self.inv.domains().index(self.domain))
        self.rb_grp = RadioButtons(self.ax_grp, [g[3:] for g in GROUPS],
                                   active=GROUPS.index(self.group))
        self.rb_lev = RadioButtons(self.ax_lev, [str(p) for p in PRESSURE_LEVELS],
                                   active=PRESSURE_LEVELS.index(500))
        self.rb_var = None
        self._build_var_radio()
        for rb in (self.rb_dom, self.rb_grp, self.rb_lev):
            _style_radio(rb)

        self.rb_dom.on_clicked(self._on_domain)
        self.rb_grp.on_clicked(self._on_group)
        self.rb_lev.on_clicked(self._on_level)

        # --- deslizador de tiempo y botones -------------------------------
        self.ax_time = self.fig.add_axes([0.235, 0.055, 0.545, 0.022])
        self.sl_time = Slider(self.ax_time, "", 0, max(1, self._ntimes() - 1),
                              valinit=0, valstep=1, color="#4878a8")
        self.sl_time.valtext.set_visible(False)
        self.sl_time.on_changed(self._on_time)

        bw, bh, by = 0.048, 0.030, 0.051
        self.btns = {}
        for i, (key, lbl) in enumerate([("prev", "◀"), ("play", "▶ anim"),
                                        ("next", "▶"), ("save", "guardar")]):
            ax = self.fig.add_axes([0.795 + i * (bw + 0.006), by, bw, bh])
            b = Button(ax, lbl, color="#f0f0f0", hovercolor="#dcdcdc")
            b.label.set_fontsize(8)
            self.btns[key] = b
        self.btns["prev"].on_clicked(lambda e: self._step(-1))
        self.btns["next"].on_clicked(lambda e: self._step(+1))
        self.btns["play"].on_clicked(lambda e: self._toggle_play())
        self.btns["save"].on_clicked(lambda e: self.save())

        # --- mapa y barra de color ----------------------------------------
        ds = self.reader.dataset(self.reader.frame(self.domain, 0).path)
        self.ax_map = make_axes(self.fig, [0.215, 0.115, 0.60, 0.80], ds)
        self.ax_cb = self.fig.add_axes([0.845, 0.155, 0.016, 0.68])

        # --- textos ---------------------------------------------------------
        self.txt_title = self.fig.text(0.215, 0.955, "", fontsize=13,
                                       color=INK, weight="semibold")
        self.txt_sub = self.fig.text(0.215, 0.928, "", fontsize=9, color=INK_SOFT)
        self.txt_note = self.fig.text(0.215, 0.028, "", fontsize=7.2,
                                      color=INK_SOFT)
        self.txt_read = self.fig.text(0.985, 0.955, "", fontsize=9, color=INK,
                                      ha="right", family="monospace")
        self.txt_help = self.fig.text(0.015, 0.105,
                                      "← →  tiempo\n↑ ↓  variable\nespacio  animar\n"
                                      "g  guardar PNG\nx  corte vertical\nc  var. del corte",
                                      fontsize=7, color=INK_SOFT, va="top")

        self.timer = self.fig.canvas.new_timer(interval=350)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_var_radio(self):
        self.ax_var.clear()
        self.ax_var.set_title("Variable", fontsize=8, color=INK, loc="left", pad=3)
        vs = group_vars(self.group)
        labels = [_wrap(v.label, 26) for v in vs]
        active = next((i for i, v in enumerate(vs) if v.id == self.spec.id), 0)
        self.rb_var = RadioButtons(self.ax_var, labels, active=active)
        _style_radio(self.rb_var, fontsize=7.4)
        self._var_ids = [v.id for v in vs]
        self.rb_var.on_clicked(self._on_var)

    # ==================================================================
    #  Estado
    # ==================================================================
    def _ntimes(self) -> int:
        return len(self.inv.frames[self.domain])

    @property
    def frames(self):
        return self.inv.frames[self.domain]

    # ==================================================================
    #  Callbacks
    # ==================================================================
    def _on_domain(self, label):
        if label == self.domain:
            return
        self.domain = label
        self.tidx = min(self.tidx, self._ntimes() - 1)
        self.sl_time.valmax = max(1, self._ntimes() - 1)
        self.sl_time.ax.set_xlim(0, self.sl_time.valmax)
        self.sl_time.set_val(min(self.tidx, self.sl_time.valmax))
        self.basemap = None
        self.redraw(full=True)

    def _on_group(self, label):
        g = next(x for x in GROUPS if x[3:] == label)
        if g == self.group:
            return
        self.group = g
        self.spec = group_vars(g)[0]
        self._build_var_radio()
        self.redraw()

    def _on_var(self, label):
        from .core import BY_ID
        vs = group_vars(self.group)
        idx = [_wrap(v.label, 26) for v in vs].index(label)
        self.spec = BY_ID[self._var_ids[idx]]
        self.redraw()

    def _on_level(self, label):
        self.level = float(label)
        if self.spec.pressure_level:
            self.redraw()

    def _on_time(self, val):
        self.tidx = int(val)
        self.redraw()

    def _step(self, d: int):
        self.tidx = (self.tidx + d) % self._ntimes()
        self.sl_time.set_val(self.tidx)

    def _toggle_play(self):
        self.playing = not self.playing
        self.btns["play"].label.set_text("‖ pausa" if self.playing else "▶ anim")
        (self.timer.start if self.playing else self.timer.stop)()

    def _tick(self):
        if self.playing:
            self._step(+1)

    def _on_key(self, ev):
        if ev.key in ("right", "shift+right"):
            self._step(6 if "shift" in ev.key else 1)
        elif ev.key in ("left", "shift+left"):
            self._step(-6 if "shift" in ev.key else -1)
        elif ev.key in ("up", "down"):
            vs = group_vars(self.group)
            i = next(i for i, v in enumerate(vs) if v.id == self.spec.id)
            i = (i + (1 if ev.key == "down" else -1)) % len(vs)
            self.rb_var.set_active(i)
        elif ev.key == " ":
            self._toggle_play()
        elif ev.key == "g":
            self.save()
        elif ev.key == "x":
            self.xs_mode = not self.xs_mode
            self.xs_pts = []
            self._set_note("Corte vertical: haz clic en el punto inicial y luego en el final."
                           if self.xs_mode else "")
        elif ev.key == "c":
            self.xs_var = XS_VARS[(XS_VARS.index(self.xs_var) + 1) % len(XS_VARS)]
            self._set_note(f"Variable del corte vertical: {self.xs_var}")

    def _latlon_from_event(self, ev):
        if ev.inaxes is not self.ax_map or ev.xdata is None:
            return None
        if HAVE_CARTOPY and hasattr(self.ax_map, "projection"):
            import cartopy.crs as ccrs
            x, y = ccrs.PlateCarree().transform_point(
                ev.xdata, ev.ydata, self.ax_map.projection)
            return (y, x)
        return (ev.ydata, ev.xdata)

    def _on_move(self, ev):
        ll = self._latlon_from_event(ev)
        if ll is None or self._data is None:
            self.txt_read.set_text("")
            return
        la, lo = ll
        lat2d, lon2d = self.reader.coords(self.domain)
        j, i = np.unravel_index(
            np.argmin((lat2d - la) ** 2 + (lon2d - lo) ** 2), lat2d.shape)
        v = self._data[j, i]
        ter = self.reader.terrain(self.domain)[j, i]
        vs = "  ---" if not np.isfinite(v) else f"{v:8.2f} {self.spec.units}"
        self.txt_read.set_text(f"{la:7.3f}, {lo:8.3f}   z={ter:5.0f} m   {vs}")
        self.fig.canvas.draw_idle()

    def _on_click(self, ev):
        if not self.xs_mode:
            return
        ll = self._latlon_from_event(ev)
        if ll is None:
            return
        self.xs_pts.append(ll)
        self._set_note(f"Punto {len(self.xs_pts)}: {ll[0]:.2f}, {ll[1]:.2f}")
        if len(self.xs_pts) == 2:
            self._draw_cross_section()
            self.xs_mode = False
            self.xs_pts = []

    # ==================================================================
    #  Dibujo
    # ==================================================================
    _data = None

    def redraw(self, full: bool = False):
        fr = self.frames[self.tidx]
        ds = self.reader.dataset(fr.path)

        if full or self.basemap is None:
            self.ax_map.remove()
            self.ax_map = make_axes(self.fig, [0.215, 0.115, 0.60, 0.80], ds)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
            self.basemap = BaseMap(self.ax_map, self.reader, self.domain)
            self._field_artists, self._barb_artist = [], None

        for a in self._field_artists:
            _remove(a)
        self._field_artists = []
        if self._barb_artist is not None:
            _remove(self._barb_artist)
            self._barb_artist = None

        lat, lon = self.reader.coords(self.domain)
        tr = self.basemap.transform if HAVE_CARTOPY else None

        try:
            data = self.reader.field(self.domain, self.tidx, self.spec,
                                     self.level if self.spec.pressure_level else None)
            self._data = data
        except Exception as e:
            self._data = None
            self.txt_title.set_text(f"{self.spec.label}  —  ERROR")
            self.txt_sub.set_text(f"{type(e).__name__}: {e}")
            self.fig.canvas.draw_idle()
            return

        mesh, arts = draw_field(self.ax_map, lon, lat, data, self.spec, tr)
        self._field_artists = arts

        if self.spec.barbs:
            ctx = Context(self.reader, self.domain, self.tidx, None)
            u, v = ctx.unstagger_wind10()
            self._barb_artist = draw_barbs(self.ax_map, lon, lat, u, v, tr)

        # --- barra de color ------------------------------------------------
        self.ax_cb.clear()
        cb = self.fig.colorbar(mesh, cax=self.ax_cb, extend=getattr(mesh, "wrfview_extend", self.spec.extend))
        cb.set_label(f"{self.spec.label} [{self.spec.units}]", fontsize=8.5,
                     color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK_SOFT)
        cb.outline.set_linewidth(0.6)

        self._update_text(fr)
        self.fig.canvas.draw_idle()

    def _update_text(self, fr):
        lvl = f"  {self.level:.0f} hPa" if self.spec.pressure_level else ""
        self.txt_title.set_text(f"{self.spec.label}{lvl}")
        t0 = self.frames[0].time
        fh = (fr.time - t0).total_seconds() / 3600.0
        loc = fr.time + LOCAL_OFFSET
        dx = self.reader.dataset(fr.path).getncattr("DX") / 1000.0
        self.txt_sub.set_text(
            f"{self.domain}  ·  {dx:.0f} km  ·  "
            f"{fr.time:%Y-%m-%d %H:%M} UTC  ({loc:%d/%m %H:%M} hora de Lima)  ·  "
            f"+{fh:.0f} h  ·  instante {self.tidx+1}/{self._ntimes()}")
        note = self.spec.note
        if self.basemap and "LANDMASK" in self.basemap.geo_source:
            note = (note + "  |  " if note else "") + \
                "costa dibujada desde LANDMASK (cartopy no descargo Natural Earth)"
        self.txt_note.set_text(note)

    def _set_note(self, msg):
        self.txt_note.set_text(msg)
        self.fig.canvas.draw_idle()

    # ==================================================================
    #  Corte vertical
    # ==================================================================
    def _draw_cross_section(self):
        if not HAVE_WRFPY:
            self._set_note("El corte vertical necesita wrf-python.")
            return
        p0, p1 = self.xs_pts
        try:
            ctx = Context(self.reader, self.domain, self.tidx, None)
            arr, xx, yy, label, units = cross_section(ctx, p0, p1, self.xs_var)
        except Exception as e:
            self._set_note(f"Corte vertical fallido: {type(e).__name__}: {e}")
            return

        fig, ax = plt.subplots(figsize=(11, 5.4))
        cmap = "nws_ref" if self.xs_var == "dbz" else "YlGnBu"
        if self.xs_var in ("tc", "wa"):
            cmap = "RdBu_r"
        m = ax.contourf(xx, yy, np.ma.masked_invalid(arr), levels=18,
                        cmap=cmap, extend="both")
        cb = fig.colorbar(m, ax=ax, pad=0.02)
        cb.set_label(f"{label} [{units}]", fontsize=9)
        cb.ax.tick_params(labelsize=7)

        # perfil de terreno bajo el transecto
        lat2d, lon2d = self.reader.coords(self.domain)
        ter = self.reader.terrain(self.domain)
        n = len(xx)
        las = np.linspace(p0[0], p1[0], n)
        los = np.linspace(p0[1], p1[1], n)
        prof = [ter[np.unravel_index(np.argmin((lat2d - a) ** 2 + (lon2d - o) ** 2),
                                     lat2d.shape)] for a, o in zip(las, los)]
        ax.fill_between(xx, 0, prof, color="#6b5b45", zorder=5)
        ax.plot(xx, prof, color="#3d3427", lw=1.0, zorder=6)

        fr = self.frames[self.tidx]
        ax.set_title(f"Corte vertical {label} — {self.domain} — "
                     f"{fr.time:%Y-%m-%d %H:%M} UTC\n"
                     f"({p0[0]:.2f}, {p0[1]:.2f}) → ({p1[0]:.2f}, {p1[1]:.2f})",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("Distancia a lo largo del transecto [km]", fontsize=9)
        ax.set_ylabel("Altura [m snm]", fontsize=9)
        ax.set_ylim(0, min(16000, float(np.nanmax(yy))))
        ax.grid(True, lw=0.3, ls=":", color="#c8c8c8")
        ax.tick_params(labelsize=8, colors=INK_SOFT)
        fig.tight_layout()
        fig.show()
        self._set_note("Corte vertical generado en una ventana aparte.")

    # ==================================================================
    def save(self):
        fr = self.frames[self.tidx]
        lvl = f"_{self.level:.0f}hPa" if self.spec.pressure_level else ""
        fn = os.path.join(
            self.outdir,
            f"{self.domain}_{self.spec.id}{lvl}_{fr.time:%Y%m%d_%H%M}.png")
        self.fig.savefig(fn, dpi=150, facecolor="white", bbox_inches="tight")
        self._set_note(f"Guardado: {fn}")
        print("Guardado:", fn)

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
def _style_radio(rb, fontsize=8):
    for lab in rb.labels:
        lab.set_fontsize(fontsize)
        lab.set_color(INK)
    try:
        rb._buttons.set_sizes([26] * len(rb.labels))
    except Exception:
        pass


def _wrap(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    cut = s.rfind(" ", 0, n)
    return s if cut < 0 else s[:cut] + "\n" + s[cut + 1:]


def _remove(a):
    try:
        a.remove()
        return
    except Exception:
        pass
    for sub in (getattr(a, "collections", None) or list(a) if isinstance(a, list) else []):
        try:
            sub.remove()
        except Exception:
            pass
