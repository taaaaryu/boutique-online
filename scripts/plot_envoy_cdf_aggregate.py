#!/usr/bin/env python3
"""
Aggregate Envoy JSONL files under a results tree and plot figures:

1) For each users count: a plot showing how the CDF of duration_ms (and optionally upstream) changes with replica count.
     Output: envoy_cdf_by_replicas_users_<users>_(duration|upstream).png

2) For each users count and replica count: a plot that shows per-run CDFs (each run a separate line).
     Output: envoy_cdf_runs_users_<users>_replicas_<replicas>_(duration|upstream).png

CLI controls (simplified):
- --compare-all : cross-service duration comparison (frontend aggregated)
- --compare-all-full : cross-service duration+upstream where frontend split into br0/brp lines
- --frontend-br : dedicated frontend br0 vs brp comparison (duration only)
- --frontend-br-full : same as --frontend-br plus upstream and network_time (duration - upstream)
    (network_time only for frontend br0/brp)

Usage:
    /home/taaaaryu/lab/microservices-demo/venv/bin/python scripts/plot_envoy_cdf_aggregate.py --root locust_results --service frontend --plot duration

Notes:
 - The script expects files with paths like:
         locust_results/.../users/<group>/<users>/replicas_<n>/run_<r>/services/<svc>/envoy_access_*.jsonl
     It extracts users, replicas and run numbers from the path. It will skip files that don't match.
 - Aggregation is across services/pods for the same users/replicas/run.
"""
import os
import re
import json
import math
from collections import defaultdict
from typing import List, Dict

import argparse
import gzip

def read_jsonl(path):
    opener = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    with opener(path, mode, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def percentile_vals(data: List[float], ps=(50,90,99)):
    if not data:
        return {p: None for p in ps}
    data_sorted = sorted(data)
    n = len(data_sorted)
    out = {}
    for p in ps:
        if n == 1:
            out[p] = data_sorted[0]
            continue
        k = (p/100) * (n-1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[p] = data_sorted[int(k)]
        else:
            out[p] = data_sorted[int(f)] * (c-k) + data_sorted[int(c)] * (k-f)
    return out

def scan_files(root: str, service: str = None):
    # regex to extract users/replicas/run/service/pod
    pattern = re.compile(r'.*/users/[^/]+/(?P<users>\d+)/replicas_(?P<replicas>\d+)/run_(?P<run>\d+)/services/(?P<service>[^/]+)/envoy_access_(?P<pod>[^.]+)\.jsonl(?:\.gz)?$')
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.startswith('envoy_access_'):
                continue
            if not (fn.endswith('.jsonl') or fn.endswith('.jsonl.gz')):
                continue
            full = os.path.join(dirpath, fn)
            m = pattern.match(full)
            if not m:
                continue
            info = m.groupdict()
            # if a specific service was requested, skip others
            if service and info.get('service') != service:
                continue
            info['path'] = full
            matches.append(info)
    return matches

def aggregate(root: str, service: str = None, split_bytes: bool = False, compute_network_time: bool = False):
    files = scan_files(root, service=service)
    # data[users][replicas][run] -> metrics buckets
    def _bucket():
        return {
            'duration': [],
            'upstream': [],
            # optional split by bytes_received
            'duration_br0': [],
            'duration_brp': [], 
            'upstream_br0': [],
            'upstream_brp': [],
            'count_br0': 0,
            'count_brp': 0,
            'network_time_br0' : [],
            'network_time_brp' : [],
            }
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(_bucket)))
    for f in files:
        svc = f.get('service')
        users = int(f['users'])
        replicas = int(f['replicas'])
        run = int(f['run'])
        path = f['path']
        for obj in read_jsonl(path):
            dur = to_float(obj.get('duration_ms'))
            us = to_float(obj.get('upstream_service_time_ms'))
            if dur is not None:
                data[users][replicas][run]['duration'].append(dur)
            if us is not None:
                data[users][replicas][run]['upstream'].append(us)

            if split_bytes:
                # classify by bytes_received
                br = obj.get('bytes_received')
                try:
                    br_val = int(br) if br is not None else None
                except Exception:
                    br_val = None

                if br_val == 0: #UserとFront間の通信
                    data[users][replicas][run]['count_br0'] += 1
                    if dur is not None:
                        data[users][replicas][run]['duration_br0'].append(dur)
                    if us is not None:
                        data[users][replicas][run]['upstream_br0'].append(us)
                    # network_time is only computed for frontend when enabled
                    if compute_network_time and svc == 'frontend' and (dur is not None and us is not None):
                        data[users][replicas][run]['network_time_br0'].append(dur - us)
                elif br_val is not None and br_val > 0: #frontと他サービス間の通信
                    data[users][replicas][run]['count_brp'] += 1
                    if dur is not None:
                        data[users][replicas][run]['duration_brp'].append(dur)
                    if us is not None:
                        data[users][replicas][run]['upstream_brp'].append(us)
                    # network_time is only computed for frontend when enabled
                    if compute_network_time and svc == 'frontend' and (dur is not None and us is not None):
                        data[users][replicas][run]['network_time_brp'].append(dur - us)
    return data

def aggregate_frontend_br_compare(root: str):
    """Aggregate only frontend samples into br0 and brp collections merged across runs and replicas per users.

    Returns dict: {'br0': {users: {'duration': [...], 'upstream': [...], 'network_time': [...] }}, 'brp': {...}}
    network_time left empty; caller can compute later if needed.
    """
    files = scan_files(root, service='frontend')
    out = {'br0': defaultdict(lambda: {'duration': [], 'upstream': [], 'network_time': []}),
           'brp': defaultdict(lambda: {'duration': [], 'upstream': [], 'network_time': []})}
    for f in files:
        users = int(f['users'])
        path = f['path']
        for obj in read_jsonl(path):
            br = obj.get('bytes_received')
            try:
                br_val = int(br) if br is not None else None
            except Exception:
                br_val = None
            dur = to_float(obj.get('duration_ms'))
            us = to_float(obj.get('upstream_service_time_ms'))
            if br_val == 0:
                if dur is not None:
                    out['br0'][users]['duration'].append(dur)
                if us is not None:
                    out['br0'][users]['upstream'].append(us)
            elif br_val is not None and br_val > 0:
                if dur is not None:
                    out['brp'][users]['duration'].append(dur)
                if us is not None:
                    out['brp'][users]['upstream'].append(us)
    return out

def compute_network_time_for_br_compare(agg):
    for cls in ('br0','brp'):
        for users, buckets in agg[cls].items():
            d = buckets['duration']
            u = buckets['upstream']
            if d and u:
                # approximate pairwise by min length
                n = min(len(d), len(u))
                buckets['network_time'] = [d[i]-u[i] for i in range(n) if (d[i] is not None and u[i] is not None)]
    return agg

def plot_frontend_br_compare(agg, outdir, include_upstream: bool, include_network_time: bool):
    import matplotlib.pyplot as plt
    base = os.path.join(outdir, 'frontend_br_compare')
    os.makedirs(base, exist_ok=True)
    # duration
    dur_dir = os.path.join(base, 'duration')
    os.makedirs(dur_dir, exist_ok=True)
    if include_upstream:
        up_dir = os.path.join(base, 'upstream')
        os.makedirs(up_dir, exist_ok=True)
    if include_network_time:
        nt_dir = os.path.join(base, 'network_time')
        os.makedirs(nt_dir, exist_ok=True)
    for users in sorted(set(list(agg['br0'].keys()) + list(agg['brp'].keys()))):
        # duration
        fig_d, ax_d = plt.subplots(figsize=(10,6))
        any_d = False
        for cls,color in [('br0','C0'),('brp','C1')]:
            vals = agg[cls].get(users, {}).get('duration', [])
            if make_cdf(ax_d, vals, label=cls, color=color):
                any_d = True
        if any_d:
            ax_d.set_title(f'frontend duration CDF users={users} (br0 vs brp)')
            ax_d.set_xlabel('ms'); ax_d.set_ylabel('CDF'); ax_d.grid(True, linestyle=':', alpha=0.6); ax_d.legend()
            outpng = os.path.join(dur_dir, f'frontend_br_compare_users_{users}_duration.png')
            fig_d.tight_layout(); fig_d.savefig(outpng); plt.close(fig_d); print('Wrote', outpng)
        else:
            plt.close(fig_d)
        if include_upstream:
            fig_u, ax_u = plt.subplots(figsize=(10,6))
            any_u = False
            for cls,color in [('br0','C0'),('brp','C1')]:
                vals = agg[cls].get(users, {}).get('upstream', [])
                if vals and make_cdf(ax_u, vals, label=cls, color=color, linestyle='--'):
                    any_u = True
            if any_u:
                ax_u.set_title(f'frontend upstream CDF users={users} (br0 vs brp)')
                ax_u.set_xlabel('ms'); ax_u.set_ylabel('CDF'); ax_u.grid(True, linestyle=':', alpha=0.6); ax_u.legend()
                outpng = os.path.join(up_dir, f'frontend_br_compare_users_{users}_upstream.png')
                fig_u.tight_layout(); fig_u.savefig(outpng); plt.close(fig_u); print('Wrote', outpng)
            else:
                plt.close(fig_u)
        if include_network_time:
            fig_nt, ax_nt = plt.subplots(figsize=(10,6))
            any_nt = False
            for cls,color in [('br0','C0'),('brp','C1')]:
                vals = agg[cls].get(users, {}).get('network_time', [])
                if vals and make_cdf(ax_nt, vals, label=cls, color=color, linestyle=':'):
                    any_nt = True
            if any_nt:
                ax_nt.set_title(f'frontend network_time CDF users={users} (br0 vs brp)')
                ax_nt.set_xlabel('ms'); ax_nt.set_ylabel('CDF'); ax_nt.grid(True, linestyle=':', alpha=0.6); ax_nt.legend()
                outpng = os.path.join(nt_dir, f'frontend_br_compare_users_{users}_network_time.png')
                fig_nt.tight_layout(); fig_nt.savefig(outpng); plt.close(fig_nt); print('Wrote', outpng)
            else:
                plt.close(fig_nt)

def make_cdf(ax, xs, label=None, **kwargs):
    import numpy as np
    xs = np.array(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if xs.size == 0:
        return False
    xs_sorted = np.sort(xs)
    y = np.arange(1, xs_sorted.size+1) / float(xs_sorted.size)
    ax.plot(xs_sorted, y, label=label, **kwargs)
    return True

def plot_by_replicas(data, outdir):
    import matplotlib.pyplot as plt
    # create per-metric subdirectories
    duration_dir = os.path.join(outdir, 'duration')
    os.makedirs(duration_dir, exist_ok=True)
    for users in sorted(data.keys()):
        replicas_keys = sorted(data[users].keys())
        if not replicas_keys:
            continue
        # Create separate plots: one for duration, optionally one for upstream
        fig_dur, ax_dur = plt.subplots(figsize=(10,6))
        any_dur = False
        for i, replicas in enumerate(replicas_keys):
            # aggregate across runs
            all_durs = []
            all_us = []
            for run in data[users][replicas].keys():
                all_durs.extend(data[users][replicas][run]['duration'])

            color = f'C{i%10}'
            if make_cdf(ax_dur, all_durs, label=f'replicas={replicas}', color=color):
                any_dur = True

        # Save duration plot if we plotted anything
        if any_dur:
            ax_dur.set_title(f'Users={users}: duration CDF by replica count')
            ax_dur.set_xlabel('ms')
            ax_dur.set_ylabel('CDF')
            ax_dur.grid(True, linestyle=':', alpha=0.6)
            ax_dur.legend()
            outpng_d = os.path.join(duration_dir, f'envoy_cdf_by_replicas_users_{users}_duration.png')
            fig_dur.tight_layout()
            fig_dur.savefig(outpng_d)
            plt.close(fig_dur)
            print('Wrote', outpng_d)
        else:
            plt.close(fig_dur)

        # Save upstream plot if we plotted any upstream samples
        # upstream removed in simplified mode


def plot_runs(data, outdir):
    import matplotlib.pyplot as plt
    # create per-metric subdirectories
    duration_dir = os.path.join(outdir, 'duration')
    os.makedirs(duration_dir, exist_ok=True)
    for users in sorted(data.keys()):
        for replicas in sorted(data[users].keys()):
            runs = sorted(data[users][replicas].keys())
            if not runs:
                continue
            # Separate duration and upstream per-run plots
            fig_dur, ax_dur = plt.subplots(figsize=(10,6))
            any_dur = False
            for i, run in enumerate(runs):
                durs = data[users][replicas][run]['duration']
                color = f'C{i%10}'
                if make_cdf(ax_dur, durs, label=f'run={run}', color=color):
                    any_dur = True

            if any_dur:
                ax_dur.set_title(f'Users={users} Replicas={replicas}: per-run duration CDFs')
                ax_dur.set_xlabel('ms')
                ax_dur.set_ylabel('CDF')
                ax_dur.grid(True, linestyle=':', alpha=0.6)
                ax_dur.legend()
                outpng_d = os.path.join(duration_dir, f'envoy_cdf_runs_users_{users}_replicas_{replicas}_duration.png')
                fig_dur.tight_layout()
                fig_dur.savefig(outpng_d)
                plt.close(fig_dur)
                print('Wrote', outpng_d)
            else:
                plt.close(fig_dur)

            # upstream removed in simplified mode

def plot_by_replicas_split(data, outdir, include_network_time: bool = False):
    """When bytes split is enabled, plot separate CDFs for bytes_received==0 and >0.

    Writes into outdir/bytes_split/(br0|brp)/(duration|upstream)/...png
    """
    import matplotlib.pyplot as plt
    base = os.path.join(outdir, 'bytes_split')
    for cls in ('br0', 'brp'):
        os.makedirs(os.path.join(base, cls, 'duration'), exist_ok=True)
        if include_network_time:
            os.makedirs(os.path.join(base, cls, 'network_time'), exist_ok=True)
    for users in sorted(data.keys()):
        replicas_keys = sorted(data[users].keys())
        if not replicas_keys:
            continue
        for cls in ('br0', 'brp'):
            fig_dur, ax_dur = plt.subplots(figsize=(10,6))
            if include_network_time:
                fig_nt, ax_nt = plt.subplots(figsize=(10,6))

            any_dur = False
            any_up = False
            any_nt = False
            for i, replicas in enumerate(replicas_keys):
                all_d = []
                all_nt = []
                for run in data[users][replicas].keys():
                    all_d.extend(data[users][replicas][run][f'duration_{cls}'])
                    if include_network_time:
                        all_nt.extend(data[users][replicas][run][f'network_time_{cls}'])
                color = f'C{i%10}'
                if make_cdf(ax_dur, all_d, label=f'replicas={replicas}', color=color):
                    any_dur = True
                if include_network_time and all_nt and make_cdf(ax_nt, all_nt, label=f'replicas={replicas}', color=color, linestyle=':'):
                    any_nt = True
            if any_dur:
                label = '0' if cls == 'br0' else '>0'
                ax_dur.set_title(f'Users={users}: duration CDF by replicas (bytes_received=={label})')
                ax_dur.set_xlabel('ms')
                ax_dur.set_ylabel('CDF')
                ax_dur.grid(True, linestyle=':', alpha=0.6)
                ax_dur.legend()
                outpng = os.path.join(base, cls, 'duration', f'envoy_cdf_by_replicas_users_{users}_duration_{cls}.png')
                fig_dur.tight_layout()
                fig_dur.savefig(outpng)
                plt.close(fig_dur)
                print('Wrote', outpng)
            else:
                plt.close(fig_dur)
            # upstream omitted in simplified mode
            if include_network_time and any_nt:
                label = '0' if cls == 'br0' else '>0'
                ax_nt.set_title(f'Users={users}: network_time CDF by replicas (bytes_received=={label})')
                ax_nt.set_xlabel('ms')
                ax_nt.set_ylabel('CDF')
                ax_nt.grid(True, linestyle=':', alpha=0.6)
                ax_nt.legend()
                outpng = os.path.join(base, cls, 'network_time', f'envoy_cdf_by_replicas_users_{users}_network_time_{cls}.png')
                fig_nt.tight_layout()
                fig_nt.savefig(outpng)
                plt.close(fig_nt)
                print('Wrote', outpng)
            elif include_network_time:
                plt.close(fig_nt)

def plot_runs_split(data, outdir, include_network_time: bool = False):
    """Per-(users,replicas) per-run CDFs for bytes split classes."""
    import matplotlib.pyplot as plt
    base = os.path.join(outdir, 'bytes_split')
    for cls in ('br0', 'brp'):
        os.makedirs(os.path.join(base, cls, 'duration'), exist_ok=True)
        if include_network_time:
            os.makedirs(os.path.join(base, cls, 'network_time'), exist_ok=True)
    for users in sorted(data.keys()):
        for replicas in sorted(data[users].keys()):
            runs = sorted(data[users][replicas].keys())
            if not runs:
                continue
            for cls in ('br0', 'brp'):
                fig_dur, ax_dur = plt.subplots(figsize=(10,6))
                if include_network_time:
                    fig_nt, ax_nt = plt.subplots(figsize=(10,6))
                any_dur = False
                any_up = False
                any_nt = False
                for i, run in enumerate(runs):
                    durs = data[users][replicas][run][f'duration_{cls}']
                    nts = data[users][replicas][run][f'network_time_{cls}'] if include_network_time else []
                    color = f'C{i%10}'
                    if make_cdf(ax_dur, durs, label=f'run={run}', color=color):
                        any_dur = True
                    if include_network_time and nts and make_cdf(ax_nt, nts, label=f'run={run}', color=color, linestyle=':'):
                        any_nt = True
                if any_dur:
                    label = '0' if cls == 'br0' else '>0'
                    ax_dur.set_title(f'Users={users} Replicas={replicas}: duration CDFs')
                    ax_dur.set_xlabel('ms')
                    ax_dur.set_ylabel('CDF')
                    ax_dur.grid(True, linestyle=':', alpha=0.6)
                    ax_dur.legend()
                    outpng = os.path.join(base, cls, 'duration', f'envoy_cdf_runs_users_{users}_replicas_{replicas}_duration_{cls}.png')
                    fig_dur.tight_layout()
                    fig_dur.savefig(outpng)
                    plt.close(fig_dur)
                    print('Wrote', outpng)
                else:
                    plt.close(fig_dur)
                # upstream omitted in simplified mode
                if include_network_time and any_nt:
                    label = '0' if cls == 'br0' else '>0'
                    ax_nt.set_title(f'Users={users} Replicas={replicas}: network_time CDFs')
                    ax_nt.set_xlabel('ms')
                    ax_nt.set_ylabel('CDF')
                    ax_nt.grid(True, linestyle=':', alpha=0.6)
                    ax_nt.legend()
                    outpng = os.path.join(base, cls, 'network_time', f'envoy_cdf_runs_users_{users}_replicas_{replicas}_network_time_{cls}.png')
                    fig_nt.tight_layout()
                    fig_nt.savefig(outpng)
                    plt.close(fig_nt)
                    print('Wrote', outpng)
                elif include_network_time:
                    plt.close(fig_nt)

def aggregate_by_service(root: str, service_filter: str = None, frontend_bytes_filter: str | None = None):
    """Aggregate samples per service -> users -> replicas (merged across runs/pods).

    If frontend_bytes_filter is 'br0' or 'brp', then for service=='frontend' we include only
    records where bytes_received==0 (br0) or >0 (brp) respectively. Otherwise, include all.

    Returns: dict: services[service][users][replicas] -> {'duration': [...], 'upstream': [...]} 
    """
    files = scan_files(root, service=None)
    services = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'duration': [], 'upstream': []})))
    for f in files:
        svc = f.get('service')
        if service_filter and svc != service_filter:
            continue
        users = int(f['users'])
        replicas = int(f['replicas'])
        path = f['path']
        for obj in read_jsonl(path):
            # determine inclusion
            include = True
            # If filtering for one class, enforce it
            if svc == 'frontend' and frontend_bytes_filter in ('br0', 'brp'):
                br = obj.get('bytes_received')
                try:
                    br_val = int(br) if br is not None else None
                except Exception:
                    br_val = None
                if frontend_bytes_filter == 'br0':
                    include = (br_val == 0)
                else:  # 'brp'
                    include = (br_val is not None and br_val > 0)

            if not include:
                continue

            dur = to_float(obj.get('duration_ms'))
            us = to_float(obj.get('upstream_service_time_ms'))
            if dur is not None:
                services[svc][users][replicas]['duration'].append(dur)
            if us is not None:
                services[svc][users][replicas]['upstream'].append(us)
    return services

def aggregate_by_service_with_frontend_split(root: str):
    """Aggregate all services; for frontend produce separate pseudo-services frontend_br0 and frontend_brp.

    Returns services dict keyed by real services plus 'frontend_br0','frontend_brp'.
    """
    files = scan_files(root, service=None)
    services = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'duration': [], 'upstream': []})))
    for f in files:
        svc = f.get('service')
        users = int(f['users'])
        replicas = int(f['replicas'])
        path = f['path']
        for obj in read_jsonl(path):
            dur = to_float(obj.get('duration_ms'))
            us = to_float(obj.get('upstream_service_time_ms'))
            if svc == 'frontend':
                br = obj.get('bytes_received')
                try:
                    br_val = int(br) if br is not None else None
                except Exception:
                    br_val = None
                target = None
                if br_val == 0:
                    target = 'frontend_br0'
                elif br_val is not None and br_val > 0:
                    target = 'frontend_brp'
                # Always still aggregate full frontend under plain 'frontend' for reference
                if dur is not None:
                    services['frontend'][users][replicas]['duration'].append(dur)
                if us is not None:
                    services['frontend'][users][replicas]['upstream'].append(us)
                if target:
                    if dur is not None:
                        services[target][users][replicas]['duration'].append(dur)
                    if us is not None:
                        services[target][users][replicas]['upstream'].append(us)
            else:
                if dur is not None:
                    services[svc][users][replicas]['duration'].append(dur)
                if us is not None:
                    services[svc][users][replicas]['upstream'].append(us)
    return services


def plot_services_compare(services_data, outdir, include_upstream: bool = False):
    """For each users and replicas, plot CDFs for all services together.

    Writes files into outdir/duration and outdir/upstream.
    """
    import matplotlib.pyplot as plt
    duration_dir = os.path.join(outdir, 'duration')
    os.makedirs(duration_dir, exist_ok=True)
    if include_upstream:
        upstream_dir = os.path.join(outdir, 'upstream')
        os.makedirs(upstream_dir, exist_ok=True)

    # collect all users & replicas present across services
    users_set = set()
    reps_set = defaultdict(set)
    for svc, svcdata in services_data.items():
        for users in svcdata.keys():
            users_set.add(users)
            for replicas in svcdata[users].keys():
                reps_set[users].add(replicas)

    services = sorted(services_data.keys())
    for users in sorted(users_set):
        for replicas in sorted(reps_set[users]):
            # duration plot
            fig_dur, ax_dur = plt.subplots(figsize=(10,6))
            any_dur = False
            for i, svc in enumerate(services):
                vals = services_data[svc].get(users, {}).get(replicas, {}).get('duration', [])
                color = f'C{i%10}'
                if make_cdf(ax_dur, vals, label=svc, color=color):
                    any_dur = True
            if any_dur:
                ax_dur.set_title(f'Users={users} Replicas={replicas}: duration CDF by service')
                ax_dur.set_xlabel('ms')
                ax_dur.set_ylabel('CDF')
                ax_dur.grid(True, linestyle=':', alpha=0.6)
                ax_dur.legend()
                outpng = os.path.join(duration_dir, f'envoy_cdf_services_users_{users}_replicas_{replicas}_duration.png')
                fig_dur.tight_layout()
                fig_dur.savefig(outpng)
                plt.close(fig_dur)
                print('Wrote', outpng)
            else:
                plt.close(fig_dur)

            # upstream plot
            if include_upstream:
                fig_up, ax_up = plt.subplots(figsize=(10,6))
                any_up = False
                for i, svc in enumerate(services):
                    vals = services_data[svc].get(users, {}).get(replicas, {}).get('upstream', [])
                    color = f'C{i%10}'
                    if vals and make_cdf(ax_up, vals, label=svc, color=color, linestyle='--'):
                        any_up = True
                if any_up:
                    ax_up.set_title(f'Users={users} Replicas={replicas}: upstream_service_time_ms CDF by service')
                    ax_up.set_xlabel('ms')
                    ax_up.set_ylabel('CDF')
                    ax_up.grid(True, linestyle=':', alpha=0.6)
                    ax_up.legend()
                    outpng = os.path.join(upstream_dir, f'envoy_cdf_services_users_{users}_replicas_{replicas}_upstream.png')
                    fig_up.tight_layout()
                    fig_up.savefig(outpng)
                    plt.close(fig_up)
                    print('Wrote', outpng)
                else:
                    plt.close(fig_up)

def print_summary(data, split_bytes: bool = False):
    for users in sorted(data.keys()):
        for replicas in sorted(data[users].keys()):
            # aggregate across runs
            all_d = []
            all_u = []
            br0 = 0
            brp = 0
            runs = sorted(data[users][replicas].keys())
            for run in runs:
                if split_bytes:
                    all_d.extend(data[users][replicas][run]['duration_br0'])
                    all_u.extend(data[users][replicas][run]['upstream_br0'])
                else:
                    all_d.extend(data[users][replicas][run]['duration'])
                    all_u.extend(data[users][replicas][run]['upstream'])
                if split_bytes:
                    br0 += data[users][replicas][run].get('count_br0', 0)
                    brp += data[users][replicas][run].get('count_brp', 0)
            p_d = percentile_vals(all_d)
            p_u = percentile_vals(all_u)
            print(f'users={users} replicas={replicas} runs={runs} samples_d={len(all_d)} samples_u={len(all_u)}')
            print(f'  duration p50/p90/p99: {p_d[50]} / {p_d[90]} / {p_d[99]}')
            print(f'  upstream p50/p90/p99: {p_u[50]} / {p_u[90]} / {p_u[99]}')
            if split_bytes:
                print(f'  bytes_received==0 count: {br0} ; >0 count: {brp}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='locust_results', help='root results directory to scan')
    parser.add_argument('--service', default=None, help='service name to filter (e.g. frontend)')
    parser.add_argument('--outdir', default='locust_results/plots_envoy', help='where to save plots')
    parser.add_argument('--compare-all', action='store_true', help='Cross-service duration comparison (aggregate frontend)')
    parser.add_argument('--compare-all-full', action='store_true', help='Cross-service duration+upstream comparison with frontend split into br0/brp')
    parser.add_argument('--frontend-br', action='store_true', help='Frontend br0 vs brp duration comparison')
    parser.add_argument('--frontend-br-full', action='store_true', help='Frontend br0 vs brp comparison including upstream and network_time')
    args = parser.parse_args()

    # determine base output directory
    base_outdir = args.outdir
    if args.service:
        output_dir = os.path.join(base_outdir, args.service)
    else:
        output_dir = base_outdir
    print("output directory:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    include_upstream = args.compare_all_full or args.frontend_br_full

    if args.compare_all or args.compare_all_full:
        if args.compare_all_full:
            services_data = aggregate_by_service_with_frontend_split(args.root)
        else:
            services_data = aggregate_by_service(args.root)
        if not services_data:
            print('No matching files found under', args.root)
            return
        plot_services_compare(services_data, output_dir, include_upstream=include_upstream)
        return

    # base per-replica/per-run charts (duration only)
    data = aggregate(args.root, service=args.service, split_bytes=False, compute_network_time=False)
    if not data:
        print('No matching files found under', args.root)
        return
    print_summary(data, split_bytes=False)
    plot_by_replicas(data, output_dir)
    plot_runs(data, output_dir)

    if args.frontend_br or args.frontend_br_full:
        agg_front = aggregate_frontend_br_compare(args.root)
        include_nt_br = args.frontend_br_full
        if include_nt_br:
            agg_front = compute_network_time_for_br_compare(agg_front)
        plot_frontend_br_compare(agg_front, output_dir, include_upstream=include_upstream, include_network_time=include_nt_br)

if __name__ == '__main__':
    main()
