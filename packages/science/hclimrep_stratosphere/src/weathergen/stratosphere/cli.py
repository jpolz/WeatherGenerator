# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Top-level CLI dispatcher for stratospheric analysis.

Usage::

    ssw-analyze <subcommand> [options]

Subcommands
-----------
polar-vortex      Zonal mean u-wind at 60°N, SSW detection.
ssw-lead-times    SSW prediction skill vs lead time.
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
        choices=["polar-vortex", "ssw-lead-times", "polar-maps"],
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
    else:
        parser.print_help()
        sys.exit(1)
