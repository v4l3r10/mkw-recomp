#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Memory census of a running build/mkw by zone, every few seconds, until it exits (Linux; on Windows memwatch_win.py
does the census, with the same arguments and files, its columns' meanings in its docstring).

Why: the process's RAM is the game's memory, the runtime's, the GPU driver's and, on an integrated GPU, the GPU's
memory too; one RSS figure does not say which to cut. The runtime labels its anonymous ranges (plat_mem_name:
"[anon:mkw guest heap]", "[anon:mkw rt ...]") and glibc labels its own with GLIBC_TUNABLES=glibc.mem.decorate_maps=1
(set it on the run), so /proc/PID/smaps tells them apart. Unlabelled anonymous memory below 2^39 is the game's (a
range remapped since it was labelled); above it, the driver's or a library's mmap. With MKW_MEM_POOL_LABELS=1 on the
run, the game's memory held by NVN pools shows as "pool 0xFLAGS" (NVN memory pool flags) instead of "guest heap".

Usage:
  uv run tools/run/memwatch.py OUT.tsv [--log build/run/NAME.log] [--every 5] [--pid PID]   (before or with the run;
                                       without --pid, the first mkw, or on Windows the mkw.exe holding the game)
  uv run tools/run/memwatch.py --report OUT.tsv                                 (peak and growth per zone)
  --detail "anon unlabelled": also OUT.detail.tsv, the 12 largest mappings of that zone per sample (t, start, size,
  rss, swap MiB, VmFlags and the mapping just below it, which tells whose an unnamed mmap is)
OUT.tsv: one row per sample and zone (MiB): t, phase (last "seq: enter" of --log), zone, rss, pss, anon, swap; zone
"=gpu" carries the amdgpu GTT and VRAM of the process (fdinfo) in rss/pss, "=system" MemAvailable and GPUReclaim.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

GUEST_LIMIT = 1 << 39  # RT_ADDRESS_LIMIT: the game's address space ends here
DRIVER_LIBS = re.compile(r"vulkan|LLVM|gallium|drm|amdgpu|radeon|nvidia|mesa|glapi|elf|zstd|expat")
MAP_LINE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \d+\s*(.*)$")
FIELDS = ("Rss", "Pss", "Anonymous", "Swap")


def zone_of(start: int, path: str) -> str:
    """The census zone of one mapping, from its label or file."""
    if path.startswith("[anon:"):
        name = path[6:-1]
        if name.startswith("mkw "):
            return name[4:]
        if name.lstrip().startswith("glibc: malloc"):
            return "malloc"
        if "stack" in name:
            return "thread stacks"
        return "anon " + re.sub(r"[:\s]*\d+$", "", name)
    if path == "[heap]":
        return "malloc"
    if path.startswith("[stack"):
        return "thread stacks"
    if path.startswith("["):
        return "kernel " + path.strip("[]")
    if not path:
        return "guest unlabelled" if start < GUEST_LIMIT else "anon unlabelled"
    if "/dev/dri" in path or path.startswith("/memfd") or "dmabuf" in path:
        return "gpu mapped"
    base = os.path.basename(path.replace(" (deleted)", ""))
    if ".so" in base:
        lib = base.split(".so")[0]
        return ("driver " if DRIVER_LIBS.search(lib) else "lib ") + lib
    return "file " + base


def smaps(pid: int, detail: str | None = None, maps: list | None = None) -> dict[str, list[int]]:
    """kB per zone: Rss, Pss, Anonymous, Swap. With detail, each mapping of that zone goes to maps as
    [start, end, rss kB, swap kB, vmflags, the path of the mapping below it]."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    zone, cur, below = None, None, ""
    with open(f"/proc/{pid}/smaps") as f:
        for line in f:
            m = MAP_LINE.match(line)
            if m:
                path = m[3].strip()
                zone = zone_of(int(m[1], 16), path)
                cur = [int(m[1], 16), int(m[2], 16), 0, 0, "", below] if detail and zone == detail else None
                if cur is not None and maps is not None:
                    maps.append(cur)
                below = path or below
                continue
            key, _, rest = line.partition(":")
            if cur is not None and key == "VmFlags":
                cur[4] = rest.strip()
            if zone and key in FIELDS:
                out[zone][FIELDS.index(key)] += int(rest.split()[0])
                if cur is not None and key in ("Rss", "Swap"):
                    cur[2 if key == "Rss" else 3] += int(rest.split()[0])
    return out


def gpu(pid: int) -> tuple[int, int]:
    """kB of amdgpu GTT and VRAM the process's DRM clients hold, each client once."""
    seen: dict[str, tuple[int, int]] = {}
    for fd in os.listdir(f"/proc/{pid}/fdinfo"):
        try:
            text = Path(f"/proc/{pid}/fdinfo/{fd}").read_text()
        except OSError:
            continue
        cid = re.search(r"^drm-client-id:\s*(\d+)", text, re.M)
        if not cid:
            continue
        gtt = re.search(r"^drm-memory-gtt:\s*(\d+)", text, re.M)
        vram = re.search(r"^drm-memory-vram:\s*(\d+)", text, re.M)
        seen[cid[1]] = (int(gtt[1]) if gtt else 0, int(vram[1]) if vram else 0)
    return sum(g for g, _ in seen.values()), sum(v for _, v in seen.values())


def system() -> tuple[int, int]:
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    kb = lambda k: int(info.get(k, "0 kB").split()[0])  # noqa: E731
    return kb("MemAvailable"), kb("GPUReclaim")


def phase_of(log: Path | None, state: dict) -> str:
    """The last "seq: enter NODE" of the run's log, read from where the previous call stopped."""
    if not log or not log.exists():
        return state.get("phase", "")
    with log.open("rb") as f:
        f.seek(state.get("pos", 0))
        for raw in f:
            m = re.search(rb"seq: enter (\S+)", raw)
            if m:
                state["phase"] = m[1].decode(errors="replace")
        state["pos"] = f.tell()
    return state.get("phase", "")


def watch(out: Path, log: Path | None, every: float, detail: str | None = None, pid: int | None = None) -> None:
    while pid is None:
        pids = subprocess.run(["pgrep", "-x", "mkw"], capture_output=True, text=True).stdout.split()
        if pids:
            pid = int(pids[0])
            break
        time.sleep(0.5)
    t0, state = time.monotonic(), {}
    det = out.with_suffix(".detail.tsv").open("w") if detail else None
    if det:
        det.write("t\tstart\tsize\trss\tswap\tvmflags\tbelow\n")
    with out.open("w") as f:
        f.write("t\tphase\tzone\trss\tpss\tanon\tswap\n")
        while True:
            maps: list = []
            try:
                zones, (gtt, vram) = smaps(pid, detail, maps), gpu(pid)
            except FileNotFoundError, ProcessLookupError:
                break
            t, phase = round(time.monotonic() - t0), phase_of(log, state)
            if det:
                for s, e, rss, swap, flags, below in sorted(maps, key=lambda x: -(x[2] + x[3]))[:12]:
                    det.write(
                        f"{t}\t{s:#x}\t{(e - s) / 1048576:.1f}\t{rss / 1024:.1f}\t{swap / 1024:.1f}\t{flags}\t{below}\n"
                    )
                det.flush()
            mib = lambda k: round(k / 1024)  # noqa: E731
            for z, v in sorted(zones.items()):
                if v[0] or v[3]:
                    f.write(f"{t}\t{phase}\t{z}\t" + "\t".join(str(mib(x)) for x in v) + "\n")
            avail, reclaim = system()
            f.write(f"{t}\t{phase}\t=gpu\t{mib(gtt)}\t{mib(vram)}\t0\t0\n")
            f.write(f"{t}\t{phase}\t=system\t{mib(avail)}\t{mib(reclaim)}\t0\t0\n")
            f.flush()
            time.sleep(every)


def report(path: Path) -> None:
    """Per zone: peak RSS (and its phase), the last sample, swap peak; totals of the process at its peak. A Windows
    census (memwatch_win.py, first line "# os=win32") counts the peak by resident memory alone: its swap column is
    memory committed but not resident, most of it no RAM at all, and the commit at that peak is printed beside it."""
    lines = path.read_text().splitlines()
    win = bool(lines) and lines[0].startswith("# os=win32")
    rows = [line.split("\t") for line in lines if not line.startswith("#")][1:]
    per: dict[str, list[tuple[int, str, int, int, int]]] = defaultdict(list)
    total: dict[int, int] = defaultdict(int)
    commit: dict[int, int] = defaultdict(int)
    phases: dict[int, str] = {}
    for t, phase, zone, rss, _pss, anon, swap in rows:
        per[zone].append((int(t), phase, int(rss), int(swap), int(anon)))
        phases[int(t)] = phase
        if not zone.startswith("="):
            total[int(t)] += int(rss) + (0 if win else int(swap))
            commit[int(t)] += int(anon)
    if not total:
        sys.exit(f"{path}: no samples")
    peak_t = max(total, key=total.get)
    what = "resident" if win else "rss+swap"
    print(f"process peak ({what}) {total[peak_t]} MiB at {peak_t} s ({phases[peak_t]}); last {total[max(total)]} MiB")
    if win:
        private = next((r[4] for r in per.get("=process", []) if r[0] == peak_t), 0)
        print(f"committed at that peak {commit[peak_t]} MiB over the zones, images and mapped files included "
              f"(the process's private commit: {private} MiB)")
    print(f"{'zone':32} {'at peak':>8} {'max':>7} {'last':>7} {'swap':>6}  max at")
    order = sorted(per, key=lambda z: -max(r[2] for r in per[z]))
    for z in order:
        s = per[z]
        at_peak = next((r[2] + r[3] for r in s if r[0] == peak_t), 0)
        top = max(s, key=lambda r: r[2])
        if top[2] < 8 and not z.startswith("="):
            continue
        print(f"{z:32} {at_peak:8} {top[2]:7} {s[-1][2]:7} {max(r[3] for r in s):6}  {top[0]} s {top[1]}")
    if win:
        print("(swap: committed, not resident; =process: rss working set, swap peak working set; =gpu: rss shared, pss "
              "dedicated; =system: rss physical available, swap commit limit; =memwatch: rss/pss census ms)")
    else:
        print("(=gpu: rss column GTT, pss VRAM; =system: rss MemAvailable, pss GPUReclaim)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--every", type=float, default=5)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--detail", help='a zone whose largest mappings go to OUT.detail.tsv, e.g. "anon unlabelled"')
    ap.add_argument("--pid", type=int, help="the process to watch")
    a = ap.parse_args()
    if a.every <= 0:
        ap.error("--every must be a positive number of seconds")
    if a.report:
        report(a.report)
    elif a.out and sys.platform == "win32":
        import memwatch_win

        memwatch_win.watch(a.out, a.log, a.every, a.detail, a.pid, phase_of)
    elif a.out:
        watch(a.out, a.log, a.every, a.detail, a.pid)
    else:
        ap.error("OUT.tsv or --report")


if __name__ == "__main__":
    main()
