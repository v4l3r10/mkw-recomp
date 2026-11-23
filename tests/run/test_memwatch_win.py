"""tools/run/memwatch_win.py, memwatch.py's census on Windows. On any platform: the guest's zones from memory.c's layout
and the log's lines (the heap, the runtime's own regions, the modules), only the runtime's reservations taken as the
guest's, private memory by what is known of it (stack, heap, guard pages, write-combined, size), images by file, the
game's process told from plat_reexec's parent, and memwatch.py's report of a Windows file (resident peaks, commit-only
zones, samples a tenth of a second apart). On Windows: an allocation made between the two queries of the address
space asked for again whole (with fakes of both); a census of a child process that commits 64 MiB and touches 16
MiB of it, creates a heap and grows it, and allocates large blocks from both heaps: the allocation committed and
resident as made, the heaps' segments and large blocks under "malloc" and "heap N", its thread's stack, the zones
against the process's private commit, a committed page never touched left out of RAM (the census reads nothing of
it), the same totals walking every region; then memwatch.py following the child until it exits and its report.
Seconds.

Usage: uv run pytest tests
"""

import contextlib
import ctypes as C
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/run"))
import memwatch  # noqa: E402
import memwatch_win as mw  # noqa: E402

MiB = 1 << 20
# lines as runtime/core/boot/loader.c and runtime/core/memory/memory.c write them (a 1.8.0 run on Windows)
LOG = b"""[host] loaded main     at 0x7000000000 size 0x37f0000: 281016 relocations, 1287 imports
[host] loaded sdk      at 0x7004400000 size 0x1010000: 37326 relocations, 8 imports
[host] mem: heap 0x202200000 bytes at 0x130000000
[host] mem: render-buffer pool: 0xb800000 bytes of the runtime's own at 0x424800000, above the game's heap
"""
BLOCK = 0x100000000


def test_guest_zones_from_the_layout_and_the_log() -> None:
    g = mw.GuestMap()
    g.feed(LOG)
    z = lambda a, base=BLOCK, size=mw.LOW_SIZE: g.zone(a, base, size, BLOCK)  # noqa: E731
    assert z(0x130000000) == z(0x130000000 + 0x202200000 - 1) == "guest heap"
    assert z(0x130000000 + 0x202200000) == "guest block other"  # past the heap the log gave
    assert z(BLOCK + mw.STACK_OFF) == "guest stack"
    assert z(0x424800000) == "rt render-buffer pool"
    assert z(0x500000000, 0x500000000, mw.ALIAS_SIZE) == "guest alias"
    assert z(0x7000001000, 0x7000000000, 0x5000000) == "guest modules"
    assert z(0x7010000000, 0x7010000000, 1 << 28) == "guest kernel"
    assert z(0x7020000000, 0x7020000000, 1 << 20) == "guest hostfn"
    edges = g.edges(BLOCK)
    assert 0x130000000 in edges and 0x424800000 + 0xB800000 in edges and BLOCK + mw.STACK_OFF in edges
    assert mw.allocation_zone(BLOCK, mw.LOW_SIZE, BLOCK) == "guest block"
    assert mw.allocation_zone(0x500000000, mw.ALIAS_SIZE, BLOCK) == "guest alias"
    assert mw.allocation_zone(0x7010000000, 1 << 28, BLOCK) == "guest kernel"


def test_guest_heap_without_its_log_line() -> None:
    # without the log's "mem: heap" line (no --log, or not read yet) the whole cap counts as heap
    g = mw.GuestMap()
    assert g.zone(BLOCK + mw.HEAP_OFF + mw.HEAP_CAP - 1, BLOCK, mw.LOW_SIZE, BLOCK) == "guest heap"
    assert g.zone(0x7000000000, 0x7000000000, 0x1000000, BLOCK) == "guest modules"


def test_log_read_as_it_grows(tmp_path: Path) -> None:
    log = tmp_path / "mkw.log"
    log.write_bytes(LOG[:60])  # the first line cut short: it waits for the next read
    g = mw.GuestMap()
    g.read(log)
    assert g.modules == []
    log.write_bytes(LOG)
    g.read(log)
    assert g.modules == [(0x7000000000, 0x37F0000), (0x7004400000, 0x1010000)] and g.heap == (0x130000000, 0x202200000)
    g.read(tmp_path / "missing.log")  # warns, the census goes on with the layout alone


def test_guest_allocations_only_where_the_runtime_puts_them() -> None:
    assert mw.is_guest(BLOCK, mw.LOW_SIZE, BLOCK) and mw.is_guest(0x500000000, mw.ALIAS_SIZE, BLOCK)
    assert mw.is_guest(0x7000000000, 0x5000000, BLOCK) and mw.is_guest(0x7010000000, 1 << 28, BLOCK)
    assert not mw.is_guest(0x7FFE0000, 1 << 16, BLOCK)  # Windows' own, below the guest's block
    assert not mw.is_guest(0x200000000, 1 << 30, BLOCK)  # a driver's, inside the guest's range
    assert not mw.is_guest(0x7006000000, 1 << 20, BLOCK)  # between the modules and the kernel, never reserved


def test_protection_parts() -> None:
    assert mw.protection_part("guest heap", 0x04) == "guest heap"
    assert mw.protection_part("guest heap", mw.PAGE_READONLY) == "guest heap read-only"
    assert mw.protection_part("guest heap", mw.PAGE_NOACCESS) == "guest heap no-access"


def test_private_zones() -> None:
    pz = mw.private_zone
    assert pz(1 << 20, 64 << 10, 0x04, True, True, None) == "thread stacks"
    assert pz(1 << 30, 1 << 29, 0x04, False, False, "heap 3") == "heap 3"
    assert pz(256 << 20, 1 << 20, 0x04, True, False, None) == "guarded stacks"
    assert pz(32 * MiB, 32 * MiB, 0x404, False, False, None) == "gpu write-combined"
    assert pz(4 * MiB, 4 * MiB, 0x204, False, False, None) == "gpu uncached"
    assert pz(1 << 30, 900 * MiB, 0x04, False, False, None) == "private unattributed 256 MiB-1 GiB"
    assert pz(2 << 30, MiB, 0x04, False, False, None) == "private unattributed >1 GiB"
    assert pz(64 << 10, 4096, 0x04, False, False, None) == "private unattributed <=64 KiB"


def test_image_zones() -> None:
    store = r"\Device\HarddiskVolume3\Windows\System32\DriverStore\FileRepository\nv_dispig.inf_amd64_1\nvoglv64.dll"
    assert mw.image_zone(store) == "driver nvoglv64"
    assert mw.image_zone(r"\Device\HarddiskVolume3\Windows\System32\kernel32.dll") == "lib system"
    assert mw.image_zone(r"\Device\HarddiskVolume4\Recomp Library\mkw-recomp\mkwgame.dll") == "image mkwgame.dll"
    assert mw.image_zone(r"\Device\HarddiskVolume4\Games\SDL3.dll") == "lib sdl3"
    assert mw.image_zone("") == "mapped section"


def test_cut_and_resident() -> None:
    assert list(mw.cut(0, 100, [10, 50, 200])) == [(0, 10), (10, 50), (50, 100)]
    assert list(mw.cut(20, 30, [10, 50])) == [(20, 30)]
    pages = [0x1000, 0x2000, 0x5000]
    assert mw.resident_in(pages, 0x1000, 0x3000) == 0x2000 and mw.resident_in(pages, 0x3000, 0x5000) == 0


def test_game_told_from_the_reexec_parent() -> None:
    # (pid, block, alias, creation time): the start screen holds neither, plat_reexec's parent the block alone
    assert mw.pick_game([(10, False, False, 1), (20, True, False, 2), (30, True, True, 3)]) == 30
    assert mw.pick_game([(30, True, True, 3), (40, True, True, 9)]) == 40  # two games: the newest
    assert mw.pick_game([(10, False, False, 1), (20, True, False, 2)]) is None


def report_of(text: str, tmp_path: Path) -> str:
    tsv = tmp_path / "w.tsv"
    tsv.write_text(text)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        memwatch.report(tsv)
    return out.getvalue()


def test_report_of_a_windows_census(tmp_path: Path) -> None:
    text = report_of(
        "# os=win32: rss resident, pss resident not shareable, anon committed, swap committed not resident\n"
        "t\tphase\tzone\trss\tpss\tanon\tswap\n"
        "0.0\t\tguest heap\t3000\t3000\t3800\t800\n0.0\t\tgpu write-combined\t0\t0\t7000\t7000\n"
        "0.0\t\t=process\t3100\t3000\t11000\t3200\n"
        "0.2\t\tguest heap\t3200\t3200\t3900\t700\n0.2\t\tgpu write-combined\t0\t0\t7500\t7500\n"
        "0.2\t\t=process\t3300\t3200\t11500\t3400\n",
        tmp_path,
    )
    # samples a fifth of a second apart stay apart; the peak is resident memory, not resident + commit not resident
    assert "process peak (resident) 3200 MiB at 0.2 s" in text
    assert "committed at that peak 11400 MiB" in text and "private commit: 11500 MiB" in text
    assert re.search(r"^guest heap\s+3200\s", text, re.M)  # at peak: resident alone
    assert re.search(r"^=process\s+3300\s", text, re.M)
    assert re.search(r"^gpu write-combined\s+0\s", text, re.M)  # commit only, still listed


def test_report_of_a_linux_census_unchanged(tmp_path: Path) -> None:
    text = report_of("t\tphase\tzone\trss\tpss\tanon\tswap\n0\ttitle\tguest heap\t3000\t2900\t3000\t100\n"
                     "5\trace\tguest heap\t3500\t3400\t3500\t200\n", tmp_path)
    assert "process peak (rss+swap) 3700 MiB at 5 s (race)" in text
    assert re.search(r"^guest heap\s+3700\s+3500\s+3500\s+200\s+5 s race", text, re.M)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' structures")
@pytest.mark.parametrize("refused_again", [False, True])
def test_an_allocation_made_between_the_two_queries(monkeypatch, capsys, refused_again: bool) -> None:
    """MemoryRegionInformation finds an address free, VirtualQueryEx then finds a view mapped there (the game maps and
    unmaps small views all the time): it is asked for again whole, with no warning; one it still refuses is walked
    region by region, with a warning, to the same numbers."""
    BASE, SIZE, INVALID = 0x10000, 0x39000, 0xC0000141
    asked = []

    def region_information(h, addr, cls, out, size, ret):
        asked.append(addr)
        if addr != BASE or len(asked) == 1 or refused_again:
            return INVALID
        r = out._obj
        r.AllocationBase, r.AllocationProtect, r.RegionType = BASE, 0x04, mw.MEM_MAPPED
        r.RegionSize = r.CommitSize = SIZE
        return 0

    def basic_information(h, addr, out, size):
        if addr >= BASE + SIZE:  # past the top of the address space
            C.set_last_error(87)
            return 0
        m = out._obj
        m.BaseAddress = m.AllocationBase = BASE
        m.RegionSize, m.State, m.Protect, m.AllocationProtect, m.Type = SIZE, mw.MEM_COMMIT, 0x04, 0x04, mw.MEM_MAPPED
        return size

    monkeypatch.setattr(mw.ntdll, "NtQueryVirtualMemory", region_information)
    monkeypatch.setattr(mw.k32, "VirtualQueryEx", basic_information)
    assert mw.allocations(0, mw.Timer()) == [(BASE, SIZE, SIZE, 0x04, mw.MEM_MAPPED)]
    assert asked[:2] == [BASE, BASE]
    warned = capsys.readouterr().err
    assert ("walked region by region" in warned) == refused_again, warned


CHILD = r"""
import ctypes as C, json, os, sys
k = C.WinDLL("kernel32")
k.VirtualAlloc.restype = C.c_void_p
k.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32]
k.HeapCreate.restype = k.HeapAlloc.restype = k.GetProcessHeap.restype = C.c_void_p
k.HeapCreate.argtypes = [C.c_uint32, C.c_size_t, C.c_size_t]
k.HeapAlloc.argtypes = [C.c_void_p, C.c_uint32, C.c_size_t]
p = k.VirtualAlloc(None, 64 << 20, 0x3000, 0x04)   # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
C.memset(p, 1, 16 << 20)
quiet = k.VirtualAlloc(None, 8 << 20, 0x3000, 0x04)   # committed, never touched
hp = k.HeapCreate(0, 0, 0)
for _ in range(64):   # 16 MiB in blocks under the heap's VirtualAlloc threshold: it grows by segments
    C.memset(k.HeapAlloc(hp, 0, 256 << 10), 2, 256 << 10)
big = k.HeapAlloc(k.GetProcessHeap(), 0, 4 << 20)   # over the threshold: a reservation of its own
big2 = k.HeapAlloc(hp, 0, 3 << 20)
C.memset(big, 3, 4 << 20)
C.memset(big2, 4, 3 << 20)
# its own pid: a virtual environment's python.exe only launches the interpreter, as a process of its own
print(json.dumps({"pid": os.getpid(), "alloc": p, "quiet": quiet, "heap": hp, "process_heap": k.GetProcessHeap(),
                  "big": big, "big2": big2}), flush=True)
sys.stdin.read()
"""


@pytest.fixture
def child():
    p = subprocess.Popen([sys.executable, "-I", "-c", CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        yield p, json.loads(p.stdout.readline())
    finally:
        if p.poll() is None:
            p.stdin.close()
            p.wait(timeout=30)


def containing(per: dict, addr: int):
    return next(v for b, v in per.items() if b <= addr < b + v[1])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' census")
def test_census_of_a_child(child) -> None:
    _, made = child
    h = mw.open_process(made["pid"])
    try:
        zones, per, (_, _, worst_ms) = mw.census(h, made["pid"], mw.GuestMap())
        _, per_again, _ = mw.census(h, made["pid"], mw.GuestMap())
        zones_r, _, _ = mw.census(h, made["pid"], mw.GuestMap(), every_region=True)
        ws, _private_ws, private, _peak = mw.process_counters(h)
    finally:
        mw.k32.CloseHandle(h)
    zone, reserved, committed, resident, _ = per[made["alloc"]]
    assert zone == "private unattributed 16-64 MiB" and reserved == committed == 64 * MiB
    assert 16 * MiB <= resident <= 17 * MiB  # what the child touched, nothing more
    # the census reads nothing of a page it has no business with: never touched, it stays out of RAM
    assert per[made["quiet"]][2] == 8 * MiB and per[made["quiet"]][3] == 0 and per_again[made["quiet"]][3] == 0
    heap = per[made["heap"]][0]
    assert heap.startswith("heap ") and heap != "heap?" and per[made["process_heap"]][0] == "malloc"
    assert containing(per, made["big"])[0] == "malloc" and containing(per, made["big2"])[0] == heap  # large blocks
    assert zones[heap][2] >= 19 * MiB  # its segments and its large block
    assert any(v[0] == "thread stacks" for v in per.values())
    assert private >= 80 * MiB and ws >= 32 * MiB and worst_ms < 50
    # every byte is in some zone: the committed private bytes match what Windows charges the process (up to the images'
    # copy-on-write pages, which are private commit but counted with their images)
    private_zones = sum(v[2] for z, v in zones.items() if not z.startswith(("image ", "driver ", "lib ", "mapped")))
    assert abs(private_zones - private) <= max(16 * MiB, private * 3 // 100)
    # walking every region gives the same totals
    for i in (0, 2):
        assert abs(sum(v[i] for v in zones_r.values()) - sum(v[i] for v in zones.values())) <= 4 * MiB


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' census")
def test_memwatch_follows_the_child_until_it_exits(child, tmp_path: Path) -> None:
    p, made = child
    out = tmp_path / "w.tsv"
    # not -I: memwatch.py imports memwatch_win from its own folder, as `uv run tools/run/memwatch.py` lets it
    watcher = subprocess.Popen([sys.executable, str(ROOT / "tools/run/memwatch.py"), str(out), "--pid",
                                str(made["pid"]), "--every", "0.2"])
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and (not out.exists() or out.read_text().count("=process") < 3):
            time.sleep(0.1)
        h = mw.open_process(made["pid"])
        ws = mw.process_counters(h)[0]
        mw.k32.CloseHandle(h)
        p.stdin.close()
        p.wait(timeout=30)
        assert watcher.wait(timeout=30) == 0  # it stops by itself once the process has exited
    finally:
        if watcher.poll() is None:
            watcher.kill()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# os=win32") and "\t=gpu\t" in text and "\tthread stacks\t" in text
    peak = int(re.search(r"process peak \(resident\) (\d+) MiB", report_of(text, tmp_path))[1])
    assert peak <= ws / MiB * 1.2 + 16  # one sample's zones, not several samples summed
