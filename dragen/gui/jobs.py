"""Job queue of the browser front end.

Every job owns a directory <workdir>/<id>/ with job.json (the Job record below), log.txt (stdout
and stderr of its worker) and the files the worker writes. Jobs wait in a queue and run in
`max_parallel` worker processes (`python -m dragen.gui.worker`). The manager is the only writer of
job.json; the status of a finished job follows from the exit code of its worker.

Directories with labels.npz and result.json but without job.json (e.g. benchmarks/results/rework)
are listed as imported runs, so existing results can be browsed as well.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

QUEUED, RUNNING, DONE, FAILED, CANCELLED, INTERRUPTED = 'queued', 'running', 'done', 'failed', 'cancelled', 'interrupted'
FINISHED = (DONE, FAILED, CANCELLED, INTERRUPTED)


@dataclass
class Job:
    id: str
    kind: str
    """'generate', 'mesh' or 'export'."""
    name: str
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    parent: Optional[str] = None
    """For a meshing or export job: the generation run it belongs to."""
    error: Optional[str] = None
    imported: bool = False

    @property
    def elapsed(self) -> Optional[float]:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict:
        out = asdict(self)
        out['elapsed'] = self.elapsed
        return out


class JobManager:
    def __init__(self, workdir, max_parallel: int = 2, python: str = sys.executable):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.python = python
        self.jobs: Dict[str, Job] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._scan()
        self._thread = threading.Thread(target=self._loop, name='dragen-jobs', daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------------------------- #

    def submit(self, kind: str, name: str, config: Optional[dict] = None, parent: Optional[str] = None,
               options: Optional[dict] = None) -> Job:
        """Create a job directory and queue the job. `config` is an RveConfig dict (generation),
        `options` the settings of an export job (written to options.json)."""
        if kind not in ('generate', 'mesh', 'export'):
            raise ValueError(f'unknown job kind {kind!r}')
        stamp = time.strftime('%Y%m%d-%H%M%S')
        job_id = f'{stamp}-{uuid.uuid4().hex[:6]}'
        job = Job(id=job_id, kind=kind, name=name or job_id, parent=parent)
        job_dir = self.path(job_id)
        job_dir.mkdir(parents=True)
        if config is not None:
            from dragen.config import RveConfig
            RveConfig.from_dict(config).save(job_dir / 'config.yaml')
        if options is not None:
            (job_dir / 'options.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
        with self._lock:
            self.jobs[job_id] = job
            self._save(job)
        return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.jobs[job_id]
            if job.status == QUEUED:
                job.status, job.finished = CANCELLED, time.time()
                self._save(job)
            elif job.status == RUNNING:
                proc = self._procs.get(job_id)
                if proc is not None:
                    _terminate(proc)
                job.status, job.finished = CANCELLED, time.time()
                self._save(job)
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            if job.status in (QUEUED, RUNNING):
                raise ValueError('cancel the job before deleting it')
            if job.imported:
                raise ValueError('imported runs are not deleted from the GUI')
            del self.jobs[job_id]
            shutil.rmtree(self.path(job_id), ignore_errors=True)

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def path(self, job_id: str) -> Path:
        return self.workdir / job_id

    def log(self, job_id: str, offset: int = 0) -> List[str]:
        path = self.path(job_id) / 'log.txt'
        if not path.exists():
            return []
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return lines[offset:]

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for job_id, proc in list(self._procs.items()):
                _terminate(proc)
                job = self.jobs[job_id]
                job.status, job.finished, job.error = INTERRUPTED, time.time(), 'server stopped'
                self._save(job)
            self._procs.clear()

    # ------------------------------------------------------------------------------------- #

    def _scan(self) -> None:
        """Load the jobs of earlier sessions; jobs that were running are marked interrupted,
        queued ones queue again."""
        for job_dir in sorted(p for p in self.workdir.iterdir() if p.is_dir()):
            meta = job_dir / 'job.json'
            if meta.exists():
                try:
                    job = Job(**json.loads(meta.read_text(encoding='utf-8')))
                except (TypeError, ValueError):
                    continue
                if job.status == RUNNING:
                    job.status, job.finished, job.error = INTERRUPTED, job.finished or time.time(), 'server stopped'
                    self._save(job)
                self.jobs[job.id] = job
            elif (job_dir / 'labels.npz').exists() and (job_dir / 'result.json').exists():
                stat = (job_dir / 'result.json').stat()
                self.jobs[job_dir.name] = Job(id=job_dir.name, kind='generate', name=job_dir.name, status=DONE,
                                              created=stat.st_mtime, imported=True)

    def _save(self, job: Job) -> None:
        if job.imported:
            return
        path = self.path(job.id) / 'job.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(asdict(job), indent=2), encoding='utf-8')
        os.replace(tmp, path)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._reap()
                self._start_queued()
            self._stop.wait(0.3)

    def _reap(self) -> None:
        for job_id, proc in list(self._procs.items()):
            code = proc.poll()
            if code is None:
                continue
            del self._procs[job_id]
            job = self.jobs.get(job_id)
            if job is None or job.status != RUNNING:
                continue
            job.finished = time.time()
            if code == 0:
                job.status = DONE
            else:
                job.status = FAILED
                tail = [l for l in self.log(job_id) if l.strip()][-1:]
                job.error = tail[0] if tail else f'worker exited with code {code}'
            self._save(job)

    def _start_queued(self) -> None:
        queued = sorted((j for j in self.jobs.values() if j.status == QUEUED), key=lambda j: j.created)
        for job in queued:
            if len(self._procs) >= self.max_parallel:
                return
            job_dir = self.path(job.id)
            log = open(job_dir / 'log.txt', 'a', encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get('PYTHONPATH', '')]),
                       PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            try:
                proc = subprocess.Popen([self.python, '-m', 'dragen.gui.worker', job.kind, str(job_dir)],
                                        cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
                                        creationflags=flags)
            except OSError as err:
                job.status, job.finished, job.error = FAILED, time.time(), str(err)
                self._save(job)
                continue
            finally:
                log.close()
            self._procs[job.id] = proc
            job.status, job.started = RUNNING, time.time()
            self._save(job)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
