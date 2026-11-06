# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Top-level CLI dispatcher for stratospheric analysis.

Usage::

    ssw-analyze <subcommand> [options]

All levels are addressed in hPa (``--levels 10 50``); the stores may spread
the column over several streams (ERA5pl 50–1000 hPa, S2Spredictors 1–70 hPa,
TEM) and the scripts merge them.  Legacy ``--channels u_29`` still works on
single-stream ERA5ml stores.

Subcommands
-----------
polar-vortex             Zonal mean u-wind at 60°N (10/50 hPa), SSW detection at 10 hPa.
ssw-lead-times           SSW prediction skill vs lead time (10 hPa).
polar-maps               Polar stereographic animations and surface impact.
vertical-structure       Height–time cross-sections of zonal wind and temperature (1–1000 hPa).
polar-cap-temperature    Area-weighted polar cap temperature by pressure level (2/10/30/50 hPa).
qg-tem                   QG-TEM diagnostics: EP flux, divergence, residual circulation;
                         compares with the predicted TEM stream when present.
precursors               SSW precursors: 100 hPa heat flux, 10 hPa wave amplitude, 500 hPa blocking.
nam                      EOF-based NAM index per level (10/50/100/500/1000 hPa).
forecast-climatology     Lead-dependent bias, RMSE, variance ratio (blurring), 10 hPa spectra.
coupling                 Polar-cap height "dripping paint", 10→1000 hPa lagged correlation,
                         thermal-wind residual.
ifs-comparison           IFS S2S ensemble onset-timing baseline (u10/t10).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="ssw-analyze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        choices=[
            "polar-vortex",
            "ssw-lead-times",
            "polar-maps",
            "vertical-structure",
            "polar-cap-temperature",
            "qg-tem",
            "precursors",
            "nam",
            "forecast-climatology",
            "coupling",
            "ifs-comparison",
        ],
        help="Analysis to run.",
    )

    # Parse only the subcommand name; pass the rest through to the script.
    args, remaining = parser.parse_known_args(argv)

    if args.subcommand == "polar-vortex":
        from weathergen.stratosphere.scripts.analyze_polar_vortex import main as _main

        _main(remaining)
    elif args.subcommand == "ssw-lead-times":
        from weathergen.stratosphere.scripts.analyze_ssw_lead_times import main as _main

        _main(remaining)
    elif args.subcommand == "polar-maps":
        from weathergen.stratosphere.scripts.analyze_polar_maps import main as _main

        _main(remaining)
    elif args.subcommand == "vertical-structure":
        from weathergen.stratosphere.scripts.analyze_vertical_structure import (
            main as _main,
        )

        _main(remaining)
    elif args.subcommand == "polar-cap-temperature":
        from weathergen.stratosphere.scripts.analyze_polar_cap_temperature import (
            main as _main,
        )

        _main(remaining)
    elif args.subcommand == "qg-tem":
        from weathergen.stratosphere.scripts.analyze_qg_tem import main as _main

        _main(remaining)
    elif args.subcommand == "precursors":
        from weathergen.stratosphere.scripts.analyze_precursors import main as _main

        _main(remaining)
    elif args.subcommand == "nam":
        from weathergen.stratosphere.scripts.analyze_nam import main as _main

        _main(remaining)
    elif args.subcommand == "forecast-climatology":
        from weathergen.stratosphere.scripts.analyze_forecast_climatology import (
            main as _main,
        )

        _main(remaining)
    elif args.subcommand == "coupling":
        from weathergen.stratosphere.scripts.analyze_coupling import main as _main

        _main(remaining)
    elif args.subcommand == "ifs-comparison":
        from weathergen.stratosphere.scripts.analyze_ifs_comparison import main as _main

        _main(remaining)
    else:
        parser.print_help()
        sys.exit(1)
