"""Code metrics of one git revision: size, structure, dependencies, import time, tests and coverage.

Each revision is checked out into its own temporary git worktree, so uncommitted local changes
never leak into the numbers.

Examples:
    venv/Scripts/python benchmarks/code_metrics.py --rev 9e475f0
    venv/Scripts/python benchmarks/code_metrics.py --rev HEAD --skip-tests

Writes benchmarks/results/code_metrics/<rev>.json and prints a summary.

Definitions:
  library code      every .py file under dragen/ and InputGenerator/ (no tests, scripts, GUI entry)
  code lines        lines that are neither blank, nor comment-only, nor part of a docstring
  module            first-level subpackage (dragen/generation, dragen/stages, ...); a .py file directly
                    in dragen/ or InputGenerator/ is a module of its own (dragen/pipeline, ...)
  external package  top-level name of an imported package that is neither standard library nor
                    part of this repository
  import time       wall time of `import <module>` in a fresh interpreter, best of 3
  test suite        `pytest -q -p no:warnings` over the revision's tests/ with coverage of the
                    library code
"""
import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'code_metrics')
LIBRARY_ROOTS = ('dragen', 'InputGenerator')
LOCAL_PACKAGES = {'dragen', 'InputGenerator', 'tests', 'benchmarks'}
ENTRY_MODULES = ('dragen', 'dragen.run', 'dragen.pipeline', 'dragen.api', 'dragen.cli')  # dragen.run: legacy, up to tag legacy-baseline


def git(*args, cwd=REPO_ROOT) -> str:
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def library_files(root: str):
    for top in LIBRARY_ROOTS:
        for dirpath, _, files in os.walk(os.path.join(root, top)):
            for f in sorted(files):
                if f.endswith('.py'):
                    yield os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, '/')


def module_of(path: str) -> str:
    parts = path.split('/')
    if len(parts) > 2:
        return f'{parts[0]}/{parts[1]}'
    return f'{parts[0]}/{parts[1][:-3]}'


def resolve_import(root: str, dotted: str):
    """Module (as in module_of) of an import inside this repository, None for external ones."""
    parts = dotted.split('.')
    if parts[0] not in LIBRARY_ROOTS:
        return None
    for n in range(len(parts), 0, -1):
        base = '/'.join(parts[:n])
        if os.path.isfile(os.path.join(root, base + '.py')):
            return module_of(base + '.py')
        if os.path.isfile(os.path.join(root, base, '__init__.py')):
            return module_of(base + '/__init__.py')
    return None


def line_kinds(source: str):
    """Sets of line numbers that are comment-only and that belong to docstrings."""
    comment, code_tokens = set(), set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            comment.add(tok.start[0])
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                              tokenize.ENDMARKER):
            code_tokens.update(range(tok.start[0], tok.end[0] + 1))
    docstring = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, 'value', None), ast.Constant) \
                    and isinstance(first.value.value, str):
                docstring.update(range(first.lineno, first.end_lineno + 1))
    return comment - code_tokens, docstring, code_tokens


def analyse_file(root: str, path: str) -> dict:
    source = open(os.path.join(root, path), encoding='utf-8', errors='replace').read()
    comment_only, docstring, code_tokens = line_kinds(source)
    code = code_tokens - docstring
    tree = ast.parse(source)
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            imports.add(n.module)
    longest = max(functions, key=lambda f: f.end_lineno - f.lineno, default=None)
    return {
        'path': path, 'module': module_of(path),
        'lines_total': len(source.splitlines()), 'code_lines': len(code),
        'docstring_lines': len(docstring), 'comment_lines': len(comment_only),
        'classes': len(classes), 'functions': len(functions),
        'longest_function': None if longest is None else
        {'name': longest.name, 'lines': longest.end_lineno - longest.lineno + 1, 'line': longest.lineno},
        'imports': sorted(imports),
    }


def import_times(root: str, python: str) -> dict:
    out = {}
    for module in ENTRY_MODULES:
        best, torch_loaded, error = None, None, None
        for _ in range(3):
            code = ('import sys, time; t = time.perf_counter(); import ' + module +
                    '; print(time.perf_counter() - t, "torch" in sys.modules)')
            res = subprocess.run([python, '-W', 'ignore', '-c', code], cwd=root, capture_output=True, text=True)
            if res.returncode != 0:
                error = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'failed'
                break
            seconds, torch = res.stdout.strip().splitlines()[-1].split()
            best = float(seconds) if best is None else min(best, float(seconds))
            torch_loaded = torch == 'True'
        out[module] = {'seconds': None if best is None else round(best, 3), 'torch_loaded': torch_loaded,
                       'error': error}
    return out


def run_tests(root: str, python: str) -> dict:
    cov_json = os.path.join(root, 'coverage.json')
    cmd = [python, '-m', 'pytest', '-q', '-p', 'no:warnings', '-p', 'no:cacheprovider',
           *[f'--cov={r}' for r in LIBRARY_ROOTS], f'--cov-report=json:{cov_json}']
    start = time.perf_counter()
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    summary = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ''
    counts = {k: int(v) for v, k in re.findall(r'(\d+) (passed|failed|xfailed|xpassed|skipped|errors?)', summary)}
    coverage = {}
    if os.path.isfile(cov_json):
        data = json.load(open(cov_json))
        coverage['total_percent'] = round(data['totals']['percent_covered'], 1)
        per_module = defaultdict(lambda: [0, 0])
        for path, f in data['files'].items():
            m = module_of(path.replace(os.sep, '/'))
            per_module[m][0] += f['summary']['covered_lines']
            per_module[m][1] += f['summary']['num_statements']
        coverage['per_module_percent'] = {m: round(100 * c / n, 1) if n else None
                                          for m, (c, n) in sorted(per_module.items())}
    return {'command': ' '.join(['pytest', '-q', '-p', 'no:warnings', *cmd[6:-1], '--cov-report=json']),
            'summary': summary, 'counts': counts, 'seconds': round(seconds, 1), 'returncode': res.returncode,
            'coverage': coverage}


def summarise(root: str, rev: str, files: list) -> dict:
    per_module = defaultdict(lambda: defaultdict(int))
    for f in files:
        m = per_module[f['module']]
        m['files'] += 1
        for k in ('code_lines', 'docstring_lines', 'comment_lines', 'classes', 'functions'):
            m[k] += f[k]
    external = sorted({i.split('.')[0] for f in files for i in f['imports']}
                      - set(sys.stdlib_module_names) - LOCAL_PACKAGES)
    internal_edges = sorted({(f['module'], target) for f in files for i in f['imports']
                             if (target := resolve_import(root, i)) not in (None, f['module'])})
    largest = max(files, key=lambda f: f['code_lines'])
    longest = max((f for f in files if f['longest_function']), key=lambda f: f['longest_function']['lines'])
    return {
        'rev': rev, 'commit': git('rev-parse', '--short', rev), 'commit_date': git('show', '-s', '--format=%ci', rev),
        'totals': {'files': len(files), 'code_lines': sum(f['code_lines'] for f in files),
                   'docstring_lines': sum(f['docstring_lines'] for f in files),
                   'comment_lines': sum(f['comment_lines'] for f in files),
                   'classes': sum(f['classes'] for f in files), 'functions': sum(f['functions'] for f in files)},
        'largest_file': {'path': largest['path'], 'code_lines': largest['code_lines']},
        'longest_function': {'path': longest['path'], **longest['longest_function']},
        'per_module': {k: dict(v) for k, v in sorted(per_module.items())},
        'external_packages': external,
        'internal_import_edges': [list(e) for e in internal_edges],
        'files': files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rev', default='HEAD')
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()

    python = sys.executable
    work = tempfile.mkdtemp(prefix='code_metrics_')
    tree = os.path.join(work, 'tree')
    git('worktree', 'add', '--detach', tree, args.rev)
    try:
        files = [analyse_file(tree, p) for p in library_files(tree)]
        result = summarise(tree, args.rev, files)
        result['import_times'] = import_times(tree, python)
        result['tests'] = None if args.skip_tests else run_tests(tree, python)
    finally:
        git('worktree', 'remove', '--force', tree)
        shutil.rmtree(work, ignore_errors=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{result['commit']}.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    printable = {k: v for k, v in result.items() if k not in ('files', 'internal_import_edges')}
    print(json.dumps(printable, indent=2))
    print(out)


if __name__ == '__main__':
    main()
