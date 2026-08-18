#!/usr/bin/env python3
"""
wrfview.py -- Visor interactivo de las salidas de WRF de la DANA Anibal.

Ejemplos
--------
  python wrfview.py --dir ~/Z_WRF/test/em_real
  python wrfview.py --dir ~/Z_WRF/test/em_real --dominio d03 --var mdbz
  python wrfview.py --dir ~/Z_WRF/test/em_real --inventario
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Visor interactivo de salidas WRF (DANA Anibal)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", "-d", default=".",
                    help="directorio con los archivos wrfout_* (por defecto: .)")
    ap.add_argument("--dominio", default=None, help="d01 | d02 | d03")
    ap.add_argument("--var", default="pcp_int", help="id de variable inicial")
    ap.add_argument("--salida", default="figuras",
                    help="carpeta donde se guardan los PNG")
    ap.add_argument("--inventario", action="store_true",
                    help="solo lista los archivos utilizables y sale")
    ap.add_argument("--variables", action="store_true",
                    help="lista el catalogo de variables y sale")
    args = ap.parse_args()

    if args.variables:
        from wrfview.core import CATALOG, GROUPS
        for g in GROUPS:
            print(f"\n{g}")
            print("-" * len(g))
            for v in (x for x in CATALOG if x.group == g):
                flags = []
                if v.pressure_level: flags.append("nivel P")
                if v.needs_prev:     flags.append("usa t-1")
                if v.barbs:          flags.append("+viento")
                fl = ("  [" + ", ".join(flags) + "]") if flags else ""
                print(f"  {v.id:12s} {v.label:38s} {v.units:9s}{fl}")
                if v.note:
                    print(f"               {v.note}")
        return 0

    from wrfview.core import Inventory
    if args.inventario:
        Inventory(args.dir)
        return 0

    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "template"):
        for bk in ("macosx", "qtagg", "tkagg"):
            try:
                matplotlib.use(bk)
                break
            except Exception:
                continue
    from wrfview.app import WRFViewer
    v = WRFViewer(args.dir, outdir=args.salida, domain=args.dominio, var=args.var)
    v.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
