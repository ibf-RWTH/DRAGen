"""Browser front end for the reworked generator.

A small HTTP server (standard library only) serves a single-page app from `static/` and a JSON API.
Generation and meshing run in separate worker processes (`dragen.gui.worker`), one directory per
job under the work directory, so the server stays responsive and past runs survive a restart.

Start with `python DRAGen_web.py` from the repository root.
"""
