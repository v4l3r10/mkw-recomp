#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Windows backend of memwatch.py: a running mkw.exe's memory by zone, from its regions (VirtualQueryEx), its working
set (QueryWorkingSet), its heaps and thread stacks (PEB, TEB) and the run's log, with the process, system and GPU
counters.

Why: memwatch.py reads /proc/PID/smaps, which Windows has no counterpart of, and plat_mem_name labels nothing there,
so nothing told which part of a Windows run's RAM and commit charge was the game's, malloc's or the GPU driver's. A
1.8.0 Grand Prix on an RTX 3080 Ti (22/11/2026) had a 9.0 GB working set and 18.9 GB of private commit: 7.9 GB of it
write-combined mappings of VRAM (Resizable BAR) that took no RAM, and 2.6 GB in four 1 GiB reservations no runtime call
makes. The process is opened with query and read rights only (never suspended, debugged or written to) and the census
runs at below-normal priority.

Zones: the guest's from the layout of runtime/core/memory/memory.c (the 14 GiB stack and heap block, the 64 GiB alias,
RT_KERNEL_BASE and RT_HOSTFN_BASE of host.h), refined by the log's "loaded NAME at", "mem: heap" and "bytes of the
runtime's own" lines, with "read-only" (frozen regions, the write watch skips them; module code) and "no-access" parts
apart; images by file ("driver NAME" for GPU driver DLLs, "lib system" for Windows'); thread stacks (each thread's TEB)
and other reservations with guard pages (fiber stacks); heap segments ("malloc" for the process heap, which the UCRT's
malloc allocates from, "heap N" for the heaps others create, "heap?" for a segment of a heap not listed: the NT heap's
segment signature 0xFFEEFFEE and owner field, an undocumented layout); private write-combined or uncached memory
("gpu write-combined", "gpu uncached": what the GPU driver maps); the rest "private unattributed" by reservation size.

Columns (MiB), as memwatch.py's: rss resident (in the working set), pss resident and not shared with another process,
anon committed, swap committed but not resident: never touched, trimmed to the page file, or not RAM at all (VRAM the
driver maps through Resizable BAR; pages a driver locks, which the working set does not list). Rows "=process": working
set, private working set, private commit (PrivateUsage), peak working set; "=system": physical memory available, 0,
commit total, commit limit; "=gpu": the process's shared (system RAM) and dedicated (VRAM) GPU memory and its GPU
commit (the GPU Process Memory counters), as Linux's GTT and VRAM; "=memwatch": milliseconds the census took walking
the regions and the working set, so its own cost shows.

Usage: uv run tools/run/memwatch.py OUT.tsv [--log LOG] [--pid PID] [--every 5]   (memwatch.py comes here on Windows)
       uv run tools/run/memwatch_win.py PID                                       (one census, printed)
The log: the game folder's logs/mkw.log, or the file its stderr goes to when a launcher captures it.
"""

from __future__ import annotations

import bisect
import ctypes as C
import re
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

MiB, GiB = 1 << 20, 1 << 30
GUEST_LIMIT = 1 << 39  # RT_ADDRESS_LIMIT (runtime/core/base/host.h)
LOW_FROM, LOW_SIZE = 0x100000000, 14 * GiB  # memory.c LOW_SEARCH_FROM, LOW_SIZE: the guest's stack and heap block
STACK_OFF, STACK_SIZE, HEAP_OFF, HEAP_CAP = 0x10000000, 0x10000000, 0x30000000, 12 * GiB  # memory.c
ALIAS_SIZE = 64 * GiB  # memory.c ALIAS_SIZE
MODULES_FROM = 0x7000000000  # the guest's modules load from here (loader.c)
KERNEL = (0x7010000000, 0x10000000)  # RT_KERNEL_BASE, RT_KERNEL_SIZE
HOSTFN = (0x7020000000, 0x100000)  # RT_HOSTFN_BASE, RT_HOSTFN_SIZE
DRIVER_DLL = re.compile(r"^(nv|_nv|amd|ati|aticfx|ig[a-z0-9]*|intel|vulkan-1|libxess|sl\.)", re.I)

MEM_COMMIT, MEM_FREE = 0x1000, 0x10000
MEM_PRIVATE, MEM_MAPPED, MEM_IMAGE = 0x20000, 0x40000, 0x1000000
PAGE_NOACCESS, PAGE_READONLY, PAGE_GUARD, PAGE_NOCACHE, PAGE_WRITECOMBINE = 0x01, 0x02, 0x100, 0x200, 0x400
HEAP_SIGNATURE = 0xFFEEFFEE  # _HEAP_SEGMENT.SegmentSignature at +0x10; the owning _HEAP at +0x28 (x64)
PEB_PROCESS_HEAP, PEB_NUMBER_OF_HEAPS, PEB_PROCESS_HEAPS = 0x30, 0xE8, 0xF0  # x64 PEB
TEB_DEALLOCATION_STACK = 0x1478  # x64 TEB: the base of the thread's stack reservation

UNATTRIBUTED = [(64 << 10, "<=64 KiB"), (MiB, "64 KiB-1 MiB"), (16 * MiB, "1-16 MiB"), (64 * MiB, "16-64 MiB"),
                (256 * MiB, "64-256 MiB"), (GiB, "256 MiB-1 GiB")]


# ---- pure parts (no Windows calls): the guest's zones and the private memory's, tested on any platform


class GuestMap:
    """The guest's zones by address: memory.c's fixed layout, refined by the run's log."""

    LOADED = re.compile(rb"loaded (\S+)\s+at (0x[0-9a-f]+) size (0x[0-9a-f]+)")
    HEAP = re.compile(rb"mem: heap (0x[0-9a-f]+) bytes at (0x[0-9a-f]+)")
    OWN = re.compile(rb"mem: ([^:\n]+): (0x[0-9a-f]+) bytes of the runtime's own at (0x[0-9a-f]+)")

    def __init__(self) -> None:
        self.modules: list[tuple[int, int]] = []
        self.heap: tuple[int, int] | None = None  # base, size: logged only when not the default size
        self.own: list[tuple[int, int, str]] = []  # base, size, name
        self.pos = 0

    def feed(self, text: bytes) -> None:
        for m in self.LOADED.finditer(text):
            self.modules.append((int(m[2], 16), int(m[3], 16)))
        for m in self.HEAP.finditer(text):
            self.heap = (int(m[2], 16), int(m[1], 16))
        for m in self.OWN.finditer(text):
            self.own.append((int(m[3], 16), int(m[2], 16), m[1].decode(errors="replace").strip()))

    def read(self, log: Path | None) -> None:
        """The lines added to the log since the last call (it is read while the run writes it)."""
        if not log or not log.exists():
            return
        with log.open("rb") as f:
            f.seek(self.pos)
            text = f.read()
        cut = text.rfind(b"\n") + 1  # a line still being written waits for the next call
        self.feed(text[:cut])
        self.pos += cut

    def edges(self, block: int | None) -> list[int]:
        """Addresses where the zone can change, for cutting regions."""
        out = [KERNEL[0], KERNEL[0] + KERNEL[1], HOSTFN[0], HOSTFN[0] + HOSTFN[1]]
        for b, s in self.modules:
            out += [b, b + s]
        for b, s, _ in self.own:
            out += [b, b + s]
        if block is not None:
            hb, hs = self.heap or (block + HEAP_OFF, HEAP_CAP)
            out += [block + STACK_OFF, block + STACK_OFF + STACK_SIZE, hb, hb + hs]
        return sorted(set(out))

    def zone(self, addr: int, alloc_base: int, alloc_reserved: int, block: int | None) -> str:
        for b, s, name in self.own:
            if b <= addr < b + s:
                return "rt " + name
        if block is not None and alloc_base == block:
            hb, hs = self.heap or (block + HEAP_OFF, HEAP_CAP)
            if hb <= addr < hb + hs:
                return "guest heap"
            if block + STACK_OFF <= addr < block + STACK_OFF + STACK_SIZE:
                return "guest stack"
            return "guest block other"
        if alloc_reserved == ALIAS_SIZE:
            return "guest alias"
        if KERNEL[0] <= addr < KERNEL[0] + KERNEL[1]:
            return "guest kernel"
        if HOSTFN[0] <= addr < HOSTFN[0] + HOSTFN[1]:
            return "guest hostfn"
        if any(b <= addr < b + s for b, s in self.modules) or (not self.modules and addr >= MODULES_FROM):
            return "guest modules"
        return "guest other"


def protection_part(zone: str, protect: int) -> str:
    """Read-only and inaccessible pages of the guest's zones apart: frozen regions (vk_mem_gen.c) and module code are
    read-only, stacks moved by MapMemory no-access (memory.c)."""
    p = protect & 0xFF
    if p == PAGE_READONLY:
        return zone + " read-only"
    if p == PAGE_NOACCESS:
        return zone + " no-access"
    return zone


def private_zone(reserved: int, committed: int, wc: int, nocache: int, guarded: bool, stack: bool,
                 heap: str | None) -> str:
    """The zone of a private allocation above the guest's range, from what is known about it, most specific first."""
    if stack:
        return "thread stacks"
    if heap:
        return heap
    if guarded:
        return "guarded stacks"
    if committed and wc * 2 >= committed:
        return "gpu write-combined"
    if committed and nocache * 2 >= committed:
        return "gpu uncached"
    for limit, name in UNATTRIBUTED:
        if reserved <= limit:
            return "private unattributed " + name
    return "private unattributed >1 GiB"


def image_zone(path: str) -> str:
    """The zone of an image or a mapped file, from its NT path (\\Device\\HarddiskVolumeN\\...)."""
    name = path.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    if not name:
        return "mapped section"
    if name in ("mkw.exe", "mkwgame.dll"):
        return "image " + name
    if DRIVER_DLL.match(name):
        return "driver " + name.rsplit(".", 1)[0]
    if "\\windows\\" in path.lower():
        return "lib system"
    return "lib " + name.rsplit(".", 1)[0]


def cut(start: int, end: int, edges: list[int]):
    """[start, end) cut at the edges inside it."""
    i = bisect.bisect_right(edges, start)
    while start < end:
        stop = min(end, edges[i]) if i < len(edges) else end
        yield start, stop
        start, i = stop, i + 1


def resident_in(pages: list[int], start: int, end: int) -> int:
    """Bytes of [start, end) in the working set, from its sorted page addresses."""
    return (bisect.bisect_left(pages, end) - bisect.bisect_left(pages, start)) * 4096


# ---- Windows

if sys.platform == "win32":
    k32 = C.WinDLL("kernel32", use_last_error=True)
    psapi = C.WinDLL("psapi", use_last_error=True)
    ntdll = C.WinDLL("ntdll")
    pdh = C.WinDLL("pdh")

    class MBI(C.Structure):  # MEMORY_BASIC_INFORMATION (x64)
        _fields_ = [("BaseAddress", C.c_uint64), ("AllocationBase", C.c_uint64), ("AllocationProtect", C.c_uint32),
                    ("PartitionId", C.c_uint16), ("RegionSize", C.c_uint64), ("State", C.c_uint32),
                    ("Protect", C.c_uint32), ("Type", C.c_uint32)]

    class PMC(C.Structure):  # PROCESS_MEMORY_COUNTERS_EX2 (EX is the first 80 bytes)
        _fields_ = [("cb", C.c_uint32), ("PageFaultCount", C.c_uint32)] + [(n, C.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage", "PrivateWorkingSetSize")] + [("SharedCommitUsage", C.c_uint64)]

    class PERF(C.Structure):  # PERFORMANCE_INFORMATION
        _fields_ = [("cb", C.c_uint32)] + [(n, C.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
            "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (n, C.c_uint32) for n in ("HandleCount", "ProcessCount", "ThreadCount")]

    class PE32(C.Structure):  # PROCESSENTRY32W
        _fields_ = [("dwSize", C.c_uint32), ("cntUsage", C.c_uint32), ("th32ProcessID", C.c_uint32),
                    ("th32DefaultHeapID", C.c_size_t), ("th32ModuleID", C.c_uint32), ("cntThreads", C.c_uint32),
                    ("th32ParentProcessID", C.c_uint32), ("pcPriClassBase", C.c_int32), ("dwFlags", C.c_uint32),
                    ("szExeFile", C.c_wchar * 260)]

    class TE32(C.Structure):  # THREADENTRY32
        _fields_ = [("dwSize", C.c_uint32), ("cntUsage", C.c_uint32), ("th32ThreadID", C.c_uint32),
                    ("th32OwnerProcessID", C.c_uint32), ("tpBasePri", C.c_int32), ("tpDeltaPri", C.c_int32),
                    ("dwFlags", C.c_uint32)]

    class PBI(C.Structure):  # PROCESS_BASIC_INFORMATION
        _fields_ = [("ExitStatus", C.c_int32), ("PebBaseAddress", C.c_uint64), ("AffinityMask", C.c_uint64),
                    ("BasePriority", C.c_int32), ("UniqueProcessId", C.c_uint64), ("InheritedFrom", C.c_uint64)]

    class TBI(C.Structure):  # THREAD_BASIC_INFORMATION
        _fields_ = [("ExitStatus", C.c_int32), ("TebBaseAddress", C.c_uint64), ("UniqueProcess", C.c_uint64),
                    ("UniqueThread", C.c_uint64), ("AffinityMask", C.c_uint64), ("Priority", C.c_int32),
                    ("BasePriority", C.c_int32)]

    class PDHValue(C.Structure):  # PDH_FMT_COUNTERVALUE with PDH_FMT_LARGE
        _fields_ = [("CStatus", C.c_uint32), ("largeValue", C.c_int64)]

    class PDHItem(C.Structure):  # PDH_FMT_COUNTERVALUE_ITEM_W
        _fields_ = [("szName", C.c_wchar_p), ("FmtValue", PDHValue)]

    k32.OpenProcess.restype = C.c_void_p
    k32.OpenProcess.argtypes = [C.c_uint32, C.c_int, C.c_uint32]
    k32.OpenThread.restype = C.c_void_p
    k32.OpenThread.argtypes = [C.c_uint32, C.c_int, C.c_uint32]
    k32.CloseHandle.argtypes = [C.c_void_p]
    k32.CreateToolhelp32Snapshot.restype = C.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [C.c_uint32, C.c_uint32]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [C.c_void_p, C.POINTER(PE32)]
    k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [C.c_void_p, C.POINTER(TE32)]
    k32.VirtualQueryEx.restype = C.c_size_t
    k32.VirtualQueryEx.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(MBI), C.c_size_t]
    k32.ReadProcessMemory.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    k32.WaitForSingleObject.restype = C.c_uint32
    k32.WaitForSingleObject.argtypes = [C.c_void_p, C.c_uint32]
    k32.GetCurrentProcess.restype = C.c_void_p
    k32.SetPriorityClass.argtypes = [C.c_void_p, C.c_uint32]
    psapi.QueryWorkingSet.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
    psapi.GetProcessMemoryInfo.argtypes = [C.c_void_p, C.POINTER(PMC), C.c_uint32]
    psapi.GetPerformanceInfo.argtypes = [C.POINTER(PERF), C.c_uint32]
    psapi.GetMappedFileNameW.argtypes = [C.c_void_p, C.c_void_p, C.c_wchar_p, C.c_uint32]
    ntdll.NtQueryInformationProcess.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_uint32, C.c_void_p]
    ntdll.NtQueryInformationThread.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_uint32, C.c_void_p]
    pdh.PdhOpenQueryW.argtypes = [C.c_wchar_p, C.c_size_t, C.POINTER(C.c_void_p)]
    pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_size_t, C.POINTER(C.c_void_p)]
    pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32),
                                                 C.POINTER(C.c_uint32), C.c_void_p]

PROCESS_QUERY_INFORMATION, PROCESS_VM_READ, SYNCHRONIZE = 0x0400, 0x0010, 0x00100000
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
INVALID_HANDLE = (1 << 64) - 1


def open_process(pid: int) -> int:
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE, False, pid)
    if not h:
        raise OSError(f"OpenProcess({pid}) failed: error {C.get_last_error()}")
    return h


def read(h: int, addr: int, n: int) -> bytes | None:
    buf, got = C.create_string_buffer(n), C.c_size_t()
    if not k32.ReadProcessMemory(h, addr, buf, n, C.byref(got)) or got.value != n:
        return None
    return buf.raw


def u64(b: bytes, off: int) -> int:
    return int.from_bytes(b[off:off + 8], "little")


def regions(h: int) -> list[tuple[int, int, int, int, int, int]]:
    """(base, size, allocation base, state, protect, type) of every region that is not free."""
    out, addr, m = [], 0, MBI()
    while addr < 1 << 47 and k32.VirtualQueryEx(h, addr, C.byref(m), C.sizeof(m)):
        if m.State != MEM_FREE:
            out.append((m.BaseAddress, m.RegionSize, m.AllocationBase, m.State, m.Protect, m.Type))
        addr = m.BaseAddress + m.RegionSize
    return out


def working_set(h: int) -> tuple[list[int], list[int]]:
    """Sorted page addresses of the working set, and of its shared pages (QueryWorkingSet: one call)."""
    n = 1 << 20
    while True:
        buf = (C.c_uint64 * (n + 1))()
        if psapi.QueryWorkingSet(h, buf, C.sizeof(buf)):
            break
        if C.get_last_error() != 24:  # ERROR_BAD_LENGTH: buf[0] has the count needed
            raise OSError(f"QueryWorkingSet failed: error {C.get_last_error()}")
        n = int(buf[0] * 1.25) + 4096
    entries = array("Q", bytes(buf)[8:8 + 8 * buf[0]])
    pages = sorted(e & ~0xFFF for e in entries)
    shared = sorted(e & ~0xFFF for e in entries if e & 0x100)
    return pages, shared


def heap_list(h: int) -> tuple[int, dict[int, int]]:
    """The process heap and every heap's index, from the PEB."""
    pbi = PBI()
    if ntdll.NtQueryInformationProcess(h, 0, C.byref(pbi), C.sizeof(pbi), None) != 0:
        return 0, {}
    peb = read(h, pbi.PebBaseAddress, PEB_PROCESS_HEAPS + 8)
    if not peb:
        return 0, {}
    n = min(int.from_bytes(peb[PEB_NUMBER_OF_HEAPS:PEB_NUMBER_OF_HEAPS + 4], "little"), 4096)
    arr = read(h, u64(peb, PEB_PROCESS_HEAPS), 8 * n) if n else b""
    heaps = [u64(arr, 8 * i) for i in range(n)] if arr else []
    return u64(peb, PEB_PROCESS_HEAP), {base: i for i, base in enumerate(heaps)}


def heap_zone(h: int, base: int, process_heap: int, heaps: dict[int, int]) -> str | None:
    """'malloc', 'heap N' or 'heap?' when the allocation starts with an NT heap segment, else None."""
    b = read(h, base, 0x30)
    if not b or int.from_bytes(b[0x10:0x14], "little") != HEAP_SIGNATURE:
        return None
    owner = u64(b, 0x28)
    if owner == process_heap:
        return "malloc"
    return f"heap {heaps[owner]}" if owner in heaps else "heap?"


def stack_bases(h: int, pid: int) -> set[int]:
    """The base of each thread's stack reservation (TEB DeallocationStack)."""
    out: set[int] = set()
    snap = k32.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
    if not snap or snap == INVALID_HANDLE:
        return out
    te = TE32(dwSize=C.sizeof(TE32))
    ok = k32.Thread32First(snap, C.byref(te))
    while ok:
        if te.th32OwnerProcessID == pid:
            ht = k32.OpenThread(THREAD_QUERY_LIMITED_INFORMATION, False, te.th32ThreadID)
            if ht:
                tbi = TBI()
                if ntdll.NtQueryInformationThread(ht, 0, C.byref(tbi), C.sizeof(tbi), None) == 0:
                    b = read(h, tbi.TebBaseAddress + TEB_DEALLOCATION_STACK, 8)
                    if b:
                        out.add(u64(b, 0))
                k32.CloseHandle(ht)
        ok = k32.Thread32Next(snap, C.byref(te))
    k32.CloseHandle(snap)
    return out


def processes(name: str = "mkw.exe") -> list[int]:
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    pids: list[int] = []
    if not snap or snap == INVALID_HANDLE:
        return pids
    pe = PE32(dwSize=C.sizeof(PE32))
    ok = k32.Process32FirstW(snap, C.byref(pe))
    while ok:
        if pe.szExeFile.lower() == name:
            pids.append(pe.th32ProcessID)
        ok = k32.Process32NextW(snap, C.byref(pe))
    k32.CloseHandle(snap)
    return pids


def is_guest(alloc_base: int, reserved: int, block: int | None) -> bool:
    """The guest's own allocations: the stack and heap block, the alias, and the modules, kernel and host functions'
    range. Windows may place its own or a driver's allocations elsewhere below GUEST_LIMIT."""
    return (alloc_base == block or (reserved == ALIAS_SIZE and alloc_base < GUEST_LIMIT)
            or MODULES_FROM <= alloc_base < HOSTFN[0] + HOSTFN[1])


def guest_block(regs) -> int | None:
    """The base of the guest's stack and heap block: the private reservation of LOW_SIZE below GUEST_LIMIT."""
    size: dict[int, int] = defaultdict(int)
    for _, n, ab, _, _, t in regs:
        if t == MEM_PRIVATE and LOW_FROM <= ab < GUEST_LIMIT:
            size[ab] += n
    return next((ab for ab, n in sorted(size.items()) if n == LOW_SIZE), None)


def find_game() -> int | None:
    """The mkw.exe that holds the guest's memory (the start screen's process does not)."""
    for pid in processes():
        try:
            h = open_process(pid)
        except OSError:
            continue
        try:
            if guest_block(regions(h)) is not None:
                return pid
        finally:
            k32.CloseHandle(h)
    return None


def process_counters(h: int) -> tuple[int, int, int, int]:
    """Working set, private working set (0 before Windows 10 22H2), private commit, peak working set."""
    m = PMC()
    for cb in (C.sizeof(PMC), PMC.PrivateWorkingSetSize.offset):
        m.cb = cb
        if psapi.GetProcessMemoryInfo(h, C.byref(m), cb):
            private_ws = m.PrivateWorkingSetSize if cb == C.sizeof(PMC) else 0
            return m.WorkingSetSize, private_ws, m.PrivateUsage, m.PeakWorkingSetSize
    raise OSError(f"GetProcessMemoryInfo failed: error {C.get_last_error()}")


def system_counters() -> tuple[int, int, int]:
    """Physical memory available, commit total, commit limit (bytes)."""
    p = PERF(cb=C.sizeof(PERF))
    if not psapi.GetPerformanceInfo(C.byref(p), p.cb):
        return 0, 0, 0
    return p.PhysicalAvailable * p.PageSize, p.CommitTotal * p.PageSize, p.CommitLimit * p.PageSize


class GpuCounters:
    """The process's GPU Process Memory counters (shared, dedicated, total committed), summed over its adapters.
    English counter names (PdhAddEnglishCounterW), so it works on a localized Windows."""

    NAMES = ("Shared Usage", "Dedicated Usage", "Total Committed")

    def __init__(self, pid: int) -> None:
        self.prefix, self.query, self.counters = f"pid_{pid}_", C.c_void_p(), []
        if pdh.PdhOpenQueryW(None, 0, C.byref(self.query)) != 0:
            self.query = None
            return
        for name in self.NAMES:
            c = C.c_void_p()
            if pdh.PdhAddEnglishCounterW(self.query, f"\\GPU Process Memory(*)\\{name}", 0, C.byref(c)) == 0:
                self.counters.append(c)
        if len(self.counters) != len(self.NAMES):
            self.query = None

    def read(self) -> tuple[int, int, int] | None:
        if not self.query or pdh.PdhCollectQueryData(self.query) != 0:
            return None
        out = []
        for c in self.counters:
            size, count = C.c_uint32(0), C.c_uint32(0)
            pdh.PdhGetFormattedCounterArrayW(c, 0x400, C.byref(size), C.byref(count), None)  # PDH_FMT_LARGE
            if not size.value:
                return None
            buf = C.create_string_buffer(size.value)
            if pdh.PdhGetFormattedCounterArrayW(c, 0x400, C.byref(size), C.byref(count), buf) != 0:
                return None
            items = C.cast(buf, C.POINTER(PDHItem))
            out.append(sum(items[i].FmtValue.largeValue for i in range(count.value)
                           if items[i].szName and items[i].szName.startswith(self.prefix)))
        return out[0], out[1], out[2]


def census(h: int, pid: int, guest: GuestMap):
    """Bytes per zone [resident, resident not shared, committed, committed not resident]; each allocation outside the
    guest's as base: (zone, reserved, committed, resident, its main protections); the census's walk and working set
    times (ms)."""
    t0 = time.perf_counter()
    regs = regions(h)
    t1 = time.perf_counter()
    pages, shared = working_set(h)
    t2 = time.perf_counter()
    block = guest_block(regs)
    edges = guest.edges(block)
    process_heap, heaps = heap_list(h)
    stacks = stack_bases(h, pid)
    allocs: dict[int, dict] = {}
    for base, size, ab, state, prot, typ in regs:
        a = allocs.setdefault(ab, {"type": typ, "reserved": 0, "committed": 0, "wc": 0, "nocache": 0,
                                   "guarded": False, "regions": []})
        a["reserved"] += size
        a["regions"].append((base, size, state, prot))
        if state == MEM_COMMIT:
            a["committed"] += size
            a["wc"] += size if prot & PAGE_WRITECOMBINE else 0
            a["nocache"] += size if prot & PAGE_NOCACHE else 0
            a["guarded"] |= bool(prot & PAGE_GUARD)
    zones: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    per: dict[int, tuple[str, int, int, int, str]] = {}
    name = C.create_unicode_buffer(1024)
    for ab, a in sorted(allocs.items()):
        if is_guest(ab, a["reserved"], block):
            whole = None  # the guest's: by address and protection, region by region
        elif a["type"] in (MEM_IMAGE, MEM_MAPPED):
            got = psapi.GetMappedFileNameW(h, ab, name, len(name))
            whole = image_zone(name.value if got else "")
        else:
            # Not read where the driver maps GPU memory (write-combined or uncached): that is no heap
            plain = a["committed"] and not a["wc"] and not a["nocache"]
            heap = heap_zone(h, ab, process_heap, heaps) if plain else None
            whole = private_zone(a["reserved"], a["committed"], a["wc"], a["nocache"], a["guarded"], ab in stacks,
                                 heap)
        res_alloc = 0
        for base, size, state, prot in a["regions"]:
            if state != MEM_COMMIT:
                continue
            pieces = cut(base, base + size, edges) if whole is None else [(base, base + size)]
            for s, e in pieces:
                z = whole or protection_part(guest.zone(s, ab, a["reserved"], block), prot)
                res = resident_in(pages, s, e)
                v = zones[z]
                v[0] += res
                v[1] += res - resident_in(shared, s, e)
                v[2] += e - s
                v[3] += e - s - res
                res_alloc += res
        if whole:
            prots: dict[str, int] = defaultdict(int)
            for _, size, state, prot in a["regions"]:
                if state == MEM_COMMIT:
                    prots[f"{prot:#x}"] += size
            mix = ",".join(f"{p}:{n // MiB}" for p, n in sorted(prots.items(), key=lambda kv: -kv[1])[:3])
            per[ab] = (whole, a["reserved"], a["committed"], res_alloc, mix)
    return zones, per, (round((t1 - t0) * 1000), round((t2 - t1) * 1000))


def watch(out: Path, log: Path | None, every: float, detail: str | None, pid: int | None, phase_of) -> None:
    """memwatch.py's loop on Windows: the same files and columns, with the meanings of this module's docstring."""
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS: the game's threads first
    while pid is None:
        pid = find_game()
        if pid is None:
            time.sleep(0.5)
    h = open_process(pid)
    gpu, guest, state, t0 = GpuCounters(pid), GuestMap(), {}, time.monotonic()
    det = out.with_suffix(".detail.tsv").open("w") if detail else None
    if det:  # memwatch.py's columns: vmflags is the allocation's main protections (hex: MiB), below stays empty
        det.write("t\tstart\tsize\trss\tswap\tvmflags\tbelow\n")
    mib = lambda b: round(b / MiB)  # noqa: E731
    with out.open("w") as f:
        f.write("# os=win32: rss resident, pss resident and not shared, anon committed, swap committed not resident\n")
        f.write("t\tphase\tzone\trss\tpss\tanon\tswap\n")
        while k32.WaitForSingleObject(h, 0) != 0:  # WAIT_OBJECT_0: the process has exited
            guest.read(log)
            try:
                zones, per, (walk_ms, ws_ms) = census(h, pid, guest)
                ws, private_ws, private, peak = process_counters(h)
            except OSError:
                break
            t, phase = round(time.monotonic() - t0), phase_of(log, state)
            for z, v in sorted(zones.items()):
                if v[0] or v[2]:
                    f.write(f"{t}\t{phase}\t{z}\t" + "\t".join(str(mib(x)) for x in v) + "\n")
            avail, commit, limit = system_counters()
            g = gpu.read() or (0, 0, 0)
            f.write(f"{t}\t{phase}\t=process\t{mib(ws)}\t{mib(private_ws)}\t{mib(private)}\t{mib(peak)}\n")
            f.write(f"{t}\t{phase}\t=system\t{mib(avail)}\t0\t{mib(commit)}\t{mib(limit)}\n")
            f.write(f"{t}\t{phase}\t=gpu\t{mib(g[0])}\t{mib(g[1])}\t{mib(g[2])}\t0\n")
            f.write(f"{t}\t{phase}\t=memwatch\t{walk_ms}\t{ws_ms}\t0\t0\n")
            f.flush()
            if det:
                mine = sorted(((b, *v[1:]) for b, v in per.items() if v[0] == detail), key=lambda r: -r[2])
                for base, reserved, committed, res, mix in mine[:12]:
                    det.write(f"{t}\t{base:#x}\t{reserved / MiB:.1f}\t{res / MiB:.1f}\t"
                              f"{(committed - res) / MiB:.1f}\t{mix}\t\n")
                det.flush()
            time.sleep(every)
    k32.CloseHandle(h)


def main() -> None:
    if sys.platform != "win32" or len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("usage (Windows): uv run tools/run/memwatch_win.py PID   (or memwatch.py OUT.tsv for a series)")
    pid = int(sys.argv[1])
    h = open_process(pid)
    zones, _, (walk_ms, ws_ms) = census(h, pid, GuestMap())
    ws, private_ws, private, peak = process_counters(h)
    g = GpuCounters(pid).read()
    print(f"{'zone':36} {'resident':>9} {'private':>8} {'commit':>8} {'not res':>8}   (MiB)")
    for z, v in sorted(zones.items(), key=lambda kv: -kv[1][2]):
        if v[2] >= MiB:
            print(f"{z:36} " + " ".join(f"{x / MiB:8,.0f}" for x in v))
    print(f"process: working set {ws / MiB:,.0f} (private {private_ws / MiB:,.0f}, peak {peak / MiB:,.0f}), private "
          f"commit {private / MiB:,.0f} MiB; census {walk_ms} + {ws_ms} ms")
    if g:
        print(f"gpu: shared {g[0] / MiB:,.0f}, dedicated {g[1] / MiB:,.0f}, committed {g[2] / MiB:,.0f} MiB")


if __name__ == "__main__":
    main()
