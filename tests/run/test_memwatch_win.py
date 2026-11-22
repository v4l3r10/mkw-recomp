"""tools/run/memwatch_win.py, memwatch.py's census on Windows. On any platform: the guest's zones from memory.c's layout
and the log's lines (the heap, the runtime's own regions, the modules), read-only and inaccessible guest pages apart,
private memory by what is known of it (stack, heap, write-combined, size), images by file, and memwatch.py's report
reading a Windows file. On Windows: a census of a child process that commits 64 MiB, touches 16 MiB of it, creates a
heap and grows it, which must find that allocation committed and resident as made, the new heap and the process heap
told apart and the thread's stack; then memwatch.py watching the child until it exits. Seconds.

Usage: uv run pytest tests
"""

import contextlib
import io
import json
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
    assert z(0x7010000000, 0x7010000000, 1 << 20) == "guest kernel"
    assert z(0x7020000000, 0x7020000000, 1 << 20) == "guest hostfn"
    edges = g.edges(BLOCK)
    assert 0x130000000 in edges and 0x424800000 + 0xB800000 in edges and BLOCK + mw.STACK_OFF in edges


def test_guest_heap_without_its_log_line() -> None:
    # memory.c logs the heap's size only when it is not the default: then the whole cap counts
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


def test_guest_allocations_only_where_the_runtime_puts_them() -> None:
    assert mw.is_guest(BLOCK, mw.LOW_SIZE, BLOCK) and mw.is_guest(0x500000000, mw.ALIAS_SIZE, BLOCK)
    assert mw.is_guest(0x7010000000, 1 << 28, BLOCK)
    assert not mw.is_guest(0x7FFE0000, 1 << 16, BLOCK)  # Windows' own, below the guest's block
    assert not mw.is_guest(0x200000000, 1 << 30, BLOCK)  # a driver's, inside the guest's range


def test_protection_parts() -> None:
    assert mw.protection_part("guest heap", 0x04) == "guest heap"
    assert mw.protection_part("guest heap", mw.PAGE_READONLY) == "guest heap read-only"
    assert mw.protection_part("guest heap", mw.PAGE_NOACCESS) == "guest heap no-access"


def test_private_zones() -> None:
    pz = mw.private_zone
    assert pz(1 << 20, 64 << 10, 0, 0, True, True, None) == "thread stacks"
    assert pz(1 << 30, 1 << 29, 0, 0, False, False, "heap 3") == "heap 3"
    assert pz(256 << 20, 1 << 20, 0, 0, True, False, None) == "guarded stacks"
    assert pz(32 * MiB, 32 * MiB, 32 * MiB, 0, False, False, None) == "gpu write-combined"
    assert pz(4 * MiB, 4 * MiB, 0, 4 * MiB, False, False, None) == "gpu uncached"
    assert pz(1 << 30, 900 * MiB, 0, 0, False, False, None) == "private unattributed 256 MiB-1 GiB"
    assert pz(2 << 30, MiB, 0, 0, False, False, None) == "private unattributed >1 GiB"
    assert pz(64 << 10, 4096, 0, 0, False, False, None) == "private unattributed <=64 KiB"


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


def test_report_of_a_windows_census(tmp_path: Path) -> None:
    tsv = tmp_path / "w.tsv"
    tsv.write_text(
        "# os=win32: rss resident, pss resident and not shared, anon committed, swap committed not resident\n"
        "t\tphase\tzone\trss\tpss\tanon\tswap\n"
        "0\t\tguest heap\t3000\t3000\t3800\t800\n0\t\tgpu write-combined\t10\t10\t7000\t6990\n"
        "0\t\t=process\t3100\t3000\t11000\t3200\n"
        "5\t\tguest heap\t3200\t3200\t3900\t700\n5\t\tgpu write-combined\t10\t10\t7500\t7490\n"
        "5\t\t=process\t3300\t3200\t11500\t3400\n"
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        memwatch.report(tsv)
    text = out.getvalue()
    # the peak is the resident memory (3 210 MiB at 5 s), not resident + committed-not-resident
    assert "process peak (resident) 3210 MiB at 5 s" in text
    assert "committed at that peak 11400 MiB" in text and "private commit: 11500 MiB" in text


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
hp = k.HeapCreate(0, 0, 0)
for _ in range(64):   # 16 MiB in blocks under the heap's VirtualAlloc threshold: it grows by segments
    C.memset(k.HeapAlloc(hp, 0, 256 << 10), 2, 256 << 10)
# its own pid: a virtual environment's python.exe only launches the interpreter, as a process of its own
print(json.dumps({"pid": os.getpid(), "alloc": p, "heap": hp, "process_heap": k.GetProcessHeap()}), flush=True)
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' census")
def test_census_of_a_child(child) -> None:
    _, made = child
    h = mw.open_process(made["pid"])
    try:
        zones, per, _ = mw.census(h, made["pid"], mw.GuestMap())
        ws, _private_ws, private, _peak = mw.process_counters(h)
    finally:
        mw.k32.CloseHandle(h)
    zone, reserved, committed, resident, _ = per[made["alloc"]]
    assert zone == "private unattributed 16-64 MiB" and reserved == committed == 64 * MiB
    assert 16 * MiB <= resident <= 17 * MiB  # what the child touched, nothing more
    assert per[made["heap"]][0].startswith("heap ") and per[made["heap"]][0] != "heap?"
    assert per[made["process_heap"]][0] == "malloc"
    assert any(v[0] == "thread stacks" for v in per.values())
    heap_zone = per[made["heap"]][0]
    assert zones[heap_zone][2] >= 16 * MiB  # its segments, all of them
    assert private >= 80 * MiB and ws >= 32 * MiB
    # every byte of the walk is in some zone: the committed private bytes match what Windows charges the process (up
    # to the images' copy-on-write pages, which are private commit but count in the images' zones)
    private_zones = sum(v[2] for z, v in zones.items() if not z.startswith(("image ", "driver ", "lib ", "mapped")))
    assert abs(private_zones - private) <= max(16 * MiB, private * 3 // 100)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' census")
def test_memwatch_follows_the_child_until_it_exits(child, tmp_path: Path) -> None:
    p, made = child
    out = tmp_path / "w.tsv"
    # not -I: memwatch.py imports memwatch_win from its own folder, as `uv run tools/run/memwatch.py` lets it
    watcher = subprocess.Popen([sys.executable, str(ROOT / "tools/run/memwatch.py"), str(out), "--pid",
                                str(made["pid"]), "--every", "0.2"])
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and (not out.exists() or out.read_text().count("=process") < 2):
            time.sleep(0.1)
        p.stdin.close()
        p.wait(timeout=30)
        assert watcher.wait(timeout=30) == 0  # it stops by itself once the process has exited
    finally:
        if watcher.poll() is None:
            watcher.kill()
    text = out.read_text()
    assert text.startswith("# os=win32") and "\t=gpu\t" in text and "\tthread stacks\t" in text
