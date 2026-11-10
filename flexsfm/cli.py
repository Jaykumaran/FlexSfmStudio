"""Command line:
  flexsfm new PROJECT --source VIDEO_OR_FOLDER [--srt S] [--csv C] [--masks M] [--step 15]
  flexsfm run PROJECT STEP [--set key=value ...]
  flexsfm all PROJECT [--from STEP] [--to STEP]     run the standard steps in order (optional steps skipped unless --with)
  flexsfm status PROJECT
  flexsfm gui [PROJECT]
  flexsfm doctor                                     check the installation (GPU, COLMAP, FlexGlue, ffmpeg, Qt) and say how to fix problems
"""
import argparse, json, os, sys
if os.environ.get('FLEXSFM_VERBOSE') != '1':
    os.environ.setdefault('GLOG_minloglevel', '1')     # COLMAP/GLOMAP: warnings and errors only (set FLEXSFM_VERBOSE=1 for everything)
import yaml
from .project import Project, STEP_ORDER, OPTIONAL, run_step, step_module


def _set(p, step, kvs):
    if not kvs: return
    mod = step_module(step); kinds = {q['key']: q['kind'] for q in mod.PARAMS}; vals = {}
    for kv in kvs:
        k, v = kv.split('=', 1)
        if k not in kinds: raise SystemExit(f'unknown setting "{k}" for {step}; known: {", ".join(kinds)}')
        vals[k] = yaml.safe_load(v) if kinds[k] not in ('str', 'path', 'file', 'dir', 'choice') else v
    p.set_params(step, vals)


def status(p):
    print(f'project {p.root}   current model: {p.state.get("model")}')
    for s in STEP_ORDER:
        st = p.step_state(s); c = st.get('checks', {})
        print(f'  {step_module(s).TITLE:32s} {p.status(s):8s} ' + (f'fail {c.get("fail", 0)} warn {c.get("warn", 0)} ok {c.get("ok", 0)}' if c else ''))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='flexsfm', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('new'); a.add_argument('project'); a.add_argument('--source', required=True)
    a.add_argument('--srt', default=''); a.add_argument('--csv', default=''); a.add_argument('--masks', default=''); a.add_argument('--step', type=int, default=15)
    a.add_argument('--gps_source', default='auto')
    r = sub.add_parser('run'); r.add_argument('project'); r.add_argument('step', choices=STEP_ORDER); r.add_argument('--set', nargs='*', default=[])
    al = sub.add_parser('all'); al.add_argument('project'); al.add_argument('--from', dest='frm', default=STEP_ORDER[0]); al.add_argument('--to', default=STEP_ORDER[-1])
    al.add_argument('--with', dest='with_', nargs='*', default=[], help='optional steps to include (markers, repair)')
    s = sub.add_parser('status'); s.add_argument('project')
    g = sub.add_parser('gui'); g.add_argument('project', nargs='?')
    d = sub.add_parser('doctor', help='check the installation'); d.add_argument('--full', action='store_true', help='also run FlexGlue once (downloads its weights)')
    a = ap.parse_args(argv)
    if a.cmd == 'new':
        p = Project.create(a.project, source=a.source, srt=a.srt, csv=a.csv, masks=a.masks, step=a.step, gps_source=a.gps_source); print('created', p.root)
    elif a.cmd == 'run':
        p = Project(a.project); _set(p, a.step, a.set); run_step(p, a.step)
    elif a.cmd == 'all':
        p = Project(a.project); i0, i1 = STEP_ORDER.index(a.frm), STEP_ORDER.index(a.to)
        for st in STEP_ORDER[i0:i1 + 1]:
            if st in OPTIONAL and st not in a.with_: continue
            run_step(p, st); p.reload()
    elif a.cmd == 'status':
        status(Project(a.project))
    elif a.cmd == 'doctor':
        from .doctor import main as dmain
        raise SystemExit(dmain(full=a.full))
    elif a.cmd == 'gui':
        from .gui.app import main as gmain
        gmain(a.project)


if __name__ == '__main__':
    main()
