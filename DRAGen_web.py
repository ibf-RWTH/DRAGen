"""Start the browser front end of the reworked DRAGen generator.

    venv/Scripts/python DRAGen_web.py [--port 8765] [--workdir <dir>] [--parallel 2] [--no-browser]

Runs are stored in the work directory, one folder per job (default: ~/DRAGen_runs). Point
--workdir at benchmarks/results/rework to browse the benchmark RVEs as imported runs.
"""
import argparse
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dragen.gui.server import serve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--workdir', default=str(Path.home() / 'DRAGen_runs'))
    parser.add_argument('--parallel', type=int, default=2, help='jobs running at the same time')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='log every HTTP request')
    args = parser.parse_args()

    server = serve(args.host, args.port, args.workdir, max_parallel=args.parallel, verbose=args.verbose)
    url = f'http://{args.host}:{server.server_address[1]}/'
    print(f'DRAGen web GUI on {url}  (work directory {Path(args.workdir).resolve()}), Ctrl+C to stop')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('stopping ...')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
