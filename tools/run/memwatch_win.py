#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# ///
"""Windows backend of memwatch.py: a running mkw.exe's memory by zone, from its allocations, its working set, its heaps
and thread stacks and the run's log, with the process, system and GPU counters.

Why: memwatch.py reads /proc/PID/smaps, which Windows has no counterpart of, and plat_mem_name labels nothing there,
so nothing told which part of a Windows run's RAM and commit charge was the game's, malloc's or the GPU driver's. A
1.8.0 Grand Prix on an RTX 3080 Ti (22/11/2026) had a 9.0 GB working set and 18.9 GB of private commit: 7.9 GB of it
write-combined mappings of VRAM (Resizable BAR) that took no RAM, and 2.6 GB in four 1 GiB reservations no runtime call
makes: the Vulkan driver's, which held 1.8 GB less at the title screen without the boot's pipeline prewarm
(MKW_PIPE_PREBUILD=0, 23/11/2026).

How, without disturbing the game: the process is opened with query and read rights only (never suspended, debugged or
written to), at below-normal priority, and only memory the census needs is read: the PEB, the TEBs and the heaps' own
lists. Each allocation is measured whole (NtQueryVirtualMemory, MemoryRegionInformation: its size and commit in
0.02 ms) and the working set (QueryWorkingSet) gives what is resident, page by page with its protection. Walking the
regions (VirtualQueryEx) would give each region's commit, but one call scans the page tables of a run of the same
protection while it holds that allocation: the game's commits, VirtualProtect and GetWriteWatch inside it wait as long
(22/11/2026, Windows 10 22H2: 13.8 ms per GiB, 55 ms over a 4 GiB run; MemoryRegionInformation 0.02 ms and no wait).
So regions are walked only in allocations with at most 16 MiB committed (stacks, to find their guard pages), or
everywhere with --regions. QueryWorkingSet holds nothing the game waits on: with 6 GiB resident one call took 158 ms
while the target's demand-zero faults and VirtualProtect took at most 0.46 ms, as idle (23/11/2026; QueryWorkingSetEx,
which would read only the pages asked for, made the target's VirtualProtect wait as long as each call, 6.6 ms per
64 Ki pages, so it is not used). The "=memwatch" row says what the census cost.

Zones: the guest's from the layout of runtime/core/memory/memory.c (the 14 GiB stack and heap block, the 64 GiB alias)
and host.h (RT_KERNEL_BASE, RT_HOSTFN_BASE), refined by the log's "loaded NAME at", "mem: heap" and "bytes of the
runtime's own" lines, with their read-only resident pages apart ("read-only": frozen regions, which the write watch
skips; module code); the commit of a guest allocation that is not resident goes to "ALLOCATION not resident" (with
--regions, to each zone, with "no-access" pages apart). Images by file ("driver NAME" for GPU driver DLLs, "lib system"
for Windows'); thread stacks (each thread's TEB) and other reservations with guard pages ("guarded stacks": fibers);
the heaps' segments and large blocks ("malloc" for the process heap, which the UCRT's malloc allocates from, "heap N"
for the heaps others create; read from each NT heap's SegmentList and VirtualAllocdBlocks, undocumented _HEAP offsets
checked on Windows 10 22H2); allocations the GPU driver maps write-combined or uncached ("gpu write-combined", "gpu
uncached"); the rest "private unattributed" by reservation size.

Columns (MiB), as memwatch.py's: rss resident (in the working set); pss resident and not shareable (Windows' private
working set: a clean page of an image or a section counts as shareable even when no other process maps it, so this is
not Linux's PSS); anon committed (of an image or a mapped file, its private copy-on-write commit only); swap committed
but not resident: never touched, trimmed from the working set (to the standby or modified list or the compression
store, still RAM, or to the page file), or not RAM at all (VRAM the driver maps through Resizable BAR; pages a driver
locks, which the working set does not list). Rows "=process": working set, private working set, private commit
(PrivateUsage), peak working set; "=system": physical memory available, 0, commit total, commit limit; "=gpu": the
process's shared (system RAM) and dedicated (VRAM) GPU memory and its GPU commit (GPU Process Memory counters), as
Linux's GTT and VRAM; "=memwatch": ms the allocations took, ms the working set took (its call and the sorting here,
memwatch's own time: 0.6-0.9 s with the game's 7-9 GB resident), ms of the longest single call into the game's
address space (what the game may have waited; at the boot, while the driver allocates, also what memwatch waited on
it: 8 ms once, 2.7 s in, 23/11/2026; then 2 ms at most), 0.

Usage: uv run tools/run/memwatch.py OUT.tsv [--log LOG] [--pid PID] [--every 5] [--regions]   (it comes here)
       uv run tools/run/memwatch_win.py PID [--regions]                                       (one census, printed)
The log: the game folder's logs/mkw.log, or the file its stderr goes to when a launcher captures it (RecompLauncher:
%APPDATA%\\com.recomplauncher.app\\game-output\\mkw-recomp.log).
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
MODULES_FROM = 0x7000000000  # the guest's modules, one reservation from here (loader.c)
KERNEL = (0x7010000000, 0x10000000)  # RT_KERNEL_BASE, RT_KERNEL_SIZE
HOSTFN = (0x7020000000, 0x100000)  # RT_HOSTFN_BASE, RT_HOSTFN_SIZE
DRIVER_DLL = re.compile(r"^(nv|_nv|amd|ati|aticfx|ig[a-z0-9]*|intel|vulkan-1|libxess|sl\.)", re.I)
WALK_CAP = 16 * MiB  # regions walked by default only in allocations with this much committed or less: one call over
# a 16 MiB run takes ~0.2 ms (13.8 ms per GiB, above)

MEM_COMMIT, MEM_FREE = 0x1000, 0x10000
MEM_PRIVATE, MEM_MAPPED, MEM_IMAGE = 0x20000, 0x40000, 0x1000000
MEM_PAGEFILE_SECTION = 0x08000000  # MemoryRegionInformation's type of a section backed by the page file (Windows 10)
PAGE_NOACCESS, PAGE_READONLY, PAGE_GUARD, PAGE_NOCACHE, PAGE_WRITECOMBINE = 0x01, 0x02, 0x100, 0x200, 0x400
STATUS_INVALID_ADDRESS = 0xC0000141
# Undocumented x64 offsets, checked on Windows 10 22H2 (10.0.19045) against GetProcessHeap, GetProcessHeaps,
# GetCurrentThreadStackLimits and HeapAlloc blocks of 256 KiB to 64 MiB
PEB_PROCESS_HEAP, PEB_NUMBER_OF_HEAPS, PEB_PROCESS_HEAPS = 0x30, 0xE8, 0xF0
TEB_DEALLOCATION_STACK = 0x1478  # the base of the thread's stack reservation
HEAP_SIGNATURE_AT, HEAP_SIGNATURE = 0x98, 0xEEFFEEFF  # _HEAP.Signature: an NT heap (a segment heap has another layout)
HEAP_VIRTUAL_BLOCKS, HEAP_SEGMENT_LIST = 0x110, 0x120  # _HEAP.VirtualAllocdBlocks, _HEAP.SegmentList (LIST_ENTRY heads)
SEGMENT_LIST_ENTRY = 0x18  # _HEAP_SEGMENT.SegmentListEntry: a segment's base is its list entry less this
LIST_CAP = 1 << 16

UNATTRIBUTED = [(64 << 10, "<=64 KiB"), (MiB, "64 KiB-1 MiB"), (16 * MiB, "1-16 MiB"), (64 * MiB, "16-64 MiB"),
                (256 * MiB, "64-256 MiB"), (GiB, "256 MiB-1 GiB")]


def warn(msg: str, _seen: set = set()) -> None:
    """Once per message, on stderr: an optional part of the census is missing, the rest goes on (_seen, shared by
    every call on purpose, is what was said already)."""
    if msg not in _seen:
        _seen.add(msg)
        print("memwatch: " + msg, file=sys.stderr, flush=True)


# ---- pure parts (no Windows calls): the guest's zones and the private memory's, tested on any platform


class GuestMap:
    """The guest's zones by address: memory.c's fixed layout, refined by the run's log."""

    LOADED = re.compile(rb"loaded (\S+)\s+at (0x[0-9a-f]+) size (0x[0-9a-f]+)")
    HEAP = re.compile(rb"mem: heap (0x[0-9a-f]+) bytes at (0x[0-9a-f]+)")
    OWN = re.compile(rb"mem: ([^:\n]+): (0x[0-9a-f]+) bytes of the runtime's own at (0x[0-9a-f]+)")

    def __init__(self) -> None:
        self.modules: list[tuple[int, int]] = []
        # base, size: memory.c logs it at each SetHeapSize that changes the size, the first included; before that
        # line (or without --log) the whole HEAP_CAP counts as heap
        self.heap: tuple[int, int] | None = None
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
        if not log:
            return
        try:
            with log.open("rb") as f:
                f.seek(self.pos)
                text = f.read()
        except OSError as e:
            warn(f"cannot read the log {log} ({e}): the guest's zones follow memory.c's layout alone")
            return
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


def allocation_zone(alloc_base: int, reserved: int, block: int | None) -> str:
    """The name of a guest allocation as a whole, for its commit that is not resident."""
    if alloc_base == block:
        return "guest block"
    if reserved == ALIAS_SIZE:
        return "guest alias"
    return {MODULES_FROM: "guest modules", KERNEL[0]: "guest kernel", HOSTFN[0]: "guest hostfn"}.get(alloc_base,
                                                                                                   "guest other")


def is_guest(alloc_base: int, reserved: int, block: int | None) -> bool:
    """The guest's own allocations: the stack and heap block, the alias, the modules', the kernel's and the host
    functions' reservations. Windows may place its own or a driver's allocations elsewhere below GUEST_LIMIT, even
    between the modules and the kernel."""
    return (alloc_base == block or alloc_base in (MODULES_FROM, KERNEL[0], HOSTFN[0])
            or (reserved == ALIAS_SIZE and alloc_base < GUEST_LIMIT))


def protection_part(zone: str, protect: int) -> str:
    """A region's read-only and inaccessible pages of a guest zone apart: frozen regions (vk_mem_gen.c) and module code
    are read-only, stacks moved by MapMemory no-access (memory.c)."""
    p = protect & 0xFF
    if p == PAGE_READONLY:
        return zone + " read-only"
    if p == PAGE_NOACCESS:
        return zone + " no-access"
    return zone


def private_zone(reserved: int, committed: int, protect: int, guarded: bool, stack: bool, heap: str | None) -> str:
    """The zone of a private allocation outside the guest's, from what is known about it, most specific first."""
    if stack:
        return "thread stacks"
    if heap:
        return heap
    if guarded:
        return "guarded stacks"
    if committed and protect & PAGE_WRITECOMBINE:
        return "gpu write-combined"
    if committed and protect & PAGE_NOCACHE:
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


def resident_in(pages, start: int, end: int) -> int:
    """Bytes of [start, end) in the working set, from its sorted page addresses."""
    return (bisect.bisect_left(pages, end) - bisect.bisect_left(pages, start)) * 4096


def pick_game(candidates: list[tuple[int, bool, bool, int]]) -> int | None:
    """Of (pid, holds the guest's block, holds its alias, creation time), the game's process: both reservations, the
    newest. plat_reexec's parent (win32_mem.c) keeps the block it reserved and waits for its child, which has both."""
    both = [c for c in candidates if c[1] and c[2]]
    return max(both, key=lambda c: c[3])[0] if both else None


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

    class MRI(C.Structure):  # MEMORY_REGION_INFORMATION (Windows 10): a whole allocation
        _fields_ = [("AllocationBase", C.c_uint64), ("AllocationProtect", C.c_uint32), ("RegionType", C.c_uint32),
                    ("RegionSize", C.c_uint64), ("CommitSize", C.c_uint64), ("PartitionId", C.c_uint64),
                    ("NodePreference", C.c_uint64)]

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
    k32.GetProcessTimes.argtypes = [C.c_void_p] + [C.POINTER(C.c_uint64)] * 4
    psapi.QueryWorkingSet.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
    psapi.GetProcessMemoryInfo.argtypes = [C.c_void_p, C.POINTER(PMC), C.c_uint32]
    psapi.GetPerformanceInfo.argtypes = [C.POINTER(PERF), C.c_uint32]
    psapi.GetMappedFileNameW.argtypes = [C.c_void_p, C.c_void_p, C.c_wchar_p, C.c_uint32]
    ntdll.NtQueryInformationProcess.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_uint32, C.c_void_p]
    ntdll.NtQueryInformationThread.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_uint32, C.c_void_p]
    ntdll.NtQueryVirtualMemory.restype = C.c_uint32
    ntdll.NtQueryVirtualMemory.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.c_void_p]
    pdh.PdhOpenQueryW.argtypes = [C.c_wchar_p, C.c_size_t, C.POINTER(C.c_void_p)]
    pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_size_t, C.POINTER(C.c_void_p)]
    pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
    pdh.PdhCloseQuery.argtypes = [C.c_void_p]
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


def exited(h: int) -> bool:
    return k32.WaitForSingleObject(h, 0) == 0  # WAIT_OBJECT_0


def read(h: int, addr: int, n: int) -> bytes | None:
    buf, got = C.create_string_buffer(n), C.c_size_t()
    if not k32.ReadProcessMemory(h, addr, buf, n, C.byref(got)) or got.value != n:
        return None
    return buf.raw


def u64(b: bytes, off: int) -> int:
    return int.from_bytes(b[off:off + 8], "little")


class Timer:
    """The longest single call into the target's address space, in ms: what one of its threads may have waited."""

    def __init__(self) -> None:
        self.worst = 0.0

    def __call__(self, fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        self.worst = max(self.worst, (time.perf_counter() - t) * 1000)
        return r


def allocations(h: int, timer: Timer) -> list[tuple[int, int, int, int, int]]:
    """(base, reserved, committed, allocation protection, type) of each allocation, from MemoryRegionInformation, with
    VirtualQueryEx only over free space (no pages to scan)."""
    out, addr, r, m = [], 0x10000, MRI(), MBI()
    while addr < 1 << 47:
        st = timer(ntdll.NtQueryVirtualMemory, h, addr, 3, C.byref(r), C.sizeof(r), None)  # MemoryRegionInformation
        if st == 0:
            t = MEM_MAPPED if r.RegionType == MEM_PAGEFILE_SECTION else r.RegionType
            out.append((r.AllocationBase, r.RegionSize, r.CommitSize, r.AllocationProtect, t))
            addr = r.AllocationBase + r.RegionSize
            continue
        if not timer(k32.VirtualQueryEx, h, addr, C.byref(m), C.sizeof(m)):
            if C.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: past the top of the address space
                break
            raise OSError(f"VirtualQueryEx failed: error {C.get_last_error()}")
        if m.State == MEM_FREE:
            addr = m.BaseAddress + m.RegionSize
            continue
        # Free when MemoryRegionInformation was asked, allocated when VirtualQueryEx was: mkw.exe maps and unmaps small
        # views all the time (23/11/2026, a 54 s run: four, 228 KiB MEM_MAPPED views, two of them unmapped again by the
        # next query). The new allocation is asked again whole; only one still refused is walked region by region
        if m.AllocationBase >= addr:
            st2 = timer(ntdll.NtQueryVirtualMemory, h, m.AllocationBase, 3, C.byref(r), C.sizeof(r), None)
            if st2 == 0:
                t = MEM_MAPPED if r.RegionType == MEM_PAGEFILE_SECTION else r.RegionType
                out.append((r.AllocationBase, r.RegionSize, r.CommitSize, r.AllocationProtect, t))
                addr = r.AllocationBase + r.RegionSize
                continue
        warn(f"MemoryRegionInformation refused {addr:#x} (status {st:#x}) and the allocation VirtualQueryEx found "
             f"there ({m.AllocationBase:#x}, type {m.Type:#x}): it is walked region by region")
        ab, typ, reserved, committed, prot = m.AllocationBase, m.Type, 0, 0, m.AllocationProtect
        while m.AllocationBase == ab and m.State != MEM_FREE:
            reserved += m.RegionSize
            committed += m.RegionSize if m.State == MEM_COMMIT else 0
            addr = m.BaseAddress + m.RegionSize
            if not timer(k32.VirtualQueryEx, h, addr, C.byref(m), C.sizeof(m)):
                break
        out.append((ab, reserved, committed, prot, typ))
    return out


def regions_of(h: int, base: int, reserved: int, timer: Timer) -> list[tuple[int, int, int, int]]:
    """(base, size, state, protect) of each region of one allocation (VirtualQueryEx: see the docstring's cost)."""
    out, addr, end, m = [], base, base + reserved, MBI()
    while addr < end:
        if not timer(k32.VirtualQueryEx, h, addr, C.byref(m), C.sizeof(m)):
            raise OSError(f"VirtualQueryEx failed: error {C.get_last_error()}")
        out.append((m.BaseAddress, m.RegionSize, m.State, m.Protect))
        addr = m.BaseAddress + m.RegionSize
    return out


def working_set(h: int):
    """Sorted page addresses of the working set: all of them, the shareable ones and the read-only ones (protection 1,
    read-only, or 3, execute-read, in the entry's low bits)."""
    n = 1 << 20
    while True:
        buf = (C.c_uint64 * (n + 1))()
        if psapi.QueryWorkingSet(h, buf, C.sizeof(buf)):
            break
        if C.get_last_error() != 24:  # ERROR_BAD_LENGTH: buf[0] has the count needed, which may grow meanwhile
            raise OSError(f"QueryWorkingSet failed: error {C.get_last_error()}")
        n = int(buf[0] * 1.25) + 4096
    entries = array("Q", bytes(buf)[8:8 + 8 * buf[0]])
    pages = array("Q", sorted(e & ~0xFFF for e in entries))
    shared = array("Q", sorted(e & ~0xFFF for e in entries if e & 0x100))
    readonly = array("Q", sorted(e & ~0xFFF for e in entries if (e & 0x7) in (1, 3)))
    return pages, shared, readonly


def heap_list(h: int) -> tuple[int, dict[int, int]] | None:
    """The process heap and every heap's index, from the PEB; None when it cannot be read."""
    pbi = PBI()
    if ntdll.NtQueryInformationProcess(h, 0, C.byref(pbi), C.sizeof(pbi), None) != 0:
        return None
    peb = read(h, pbi.PebBaseAddress, PEB_PROCESS_HEAPS + 8)
    if not peb:
        return None
    n = min(int.from_bytes(peb[PEB_NUMBER_OF_HEAPS:PEB_NUMBER_OF_HEAPS + 4], "little"), 4096)
    arr = read(h, u64(peb, PEB_PROCESS_HEAPS), 8 * n) if n else b""
    if n and not arr:
        return None
    return u64(peb, PEB_PROCESS_HEAP), {u64(arr, 8 * i): i for i in range(n)}


def list_walk(h: int, head: int) -> list[int] | None:
    """The entries of a LIST_ENTRY list the target wrote, each Blink checked; None if the list does not hold."""
    out, prev = [], head
    first = read(h, head, 16)
    if not first:
        return None
    cur = u64(first, 0)
    while cur != head:
        e = read(h, cur, 16)
        if not e or u64(e, 8) != prev or len(out) >= LIST_CAP:
            return None
        out.append(cur)
        prev, cur = cur, u64(e, 0)
    return out


def heap_owners(h: int, heaps: tuple[int, dict[int, int]], alloc_of) -> dict[int, str]:
    """Allocation base -> 'malloc' or 'heap N' for each NT heap's segments and large blocks (its own lists only)."""
    process_heap, index = heaps
    out: dict[int, str] = {}
    for hb, i in index.items():
        sig = read(h, hb + HEAP_SIGNATURE_AT, 4)
        if not sig or int.from_bytes(sig, "little") != HEAP_SIGNATURE:
            continue  # a segment heap, or not readable
        label = "malloc" if hb == process_heap else f"heap {i}"
        segs = list_walk(h, hb + HEAP_SEGMENT_LIST)
        big = list_walk(h, hb + HEAP_VIRTUAL_BLOCKS)
        if segs is None or big is None:
            warn(f"heap {i} at {hb:#x}: its lists do not hold (another Windows layout?), its memory stays unattributed")
            continue
        for addr in [s - SEGMENT_LIST_ENTRY for s in segs] + big:
            base = alloc_of(addr)
            if base is not None:
                out[base] = label
    return out


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
            if ht:  # a thread that exited since the snapshot is skipped
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


def guest_block(allocs) -> int | None:
    """The base of the guest's stack and heap block: the private reservation of LOW_SIZE below GUEST_LIMIT."""
    return next((b for b, n, _, _, t in allocs if t == MEM_PRIVATE and n == LOW_SIZE and LOW_FROM <= b < GUEST_LIMIT),
                None)


def find_game() -> int | None:
    """The mkw.exe that holds the guest's memory (not the start screen's process, nor plat_reexec's parent)."""
    candidates = []
    for pid in processes():
        try:
            h = open_process(pid)
        except OSError as e:
            warn(f"mkw.exe PID {pid} cannot be opened ({e}): if the game runs elevated, run memwatch elevated too")
            continue
        try:
            allocs = allocations(h, Timer())
            times = [C.c_uint64() for _ in range(4)]
            k32.GetProcessTimes(h, *(C.byref(t) for t in times))
            block = guest_block(allocs) is not None
            alias = any(n == ALIAS_SIZE and t == MEM_PRIVATE and b < GUEST_LIMIT for b, n, _, _, t in allocs)
            candidates.append((pid, block, alias, times[0].value))
        except OSError as e:
            warn(f"mkw.exe PID {pid} cannot be measured ({e})")
        finally:
            k32.CloseHandle(h)
    return pick_game(candidates)


def process_counters(h: int) -> tuple[int, int, int, int]:
    """Working set, private working set (0 where Windows has no PROCESS_MEMORY_COUNTERS_EX2), private commit, peak
    working set."""
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
        warn("GetPerformanceInfo failed: =system written as 0")
        return 0, 0, 0
    return p.PhysicalAvailable * p.PageSize, p.CommitTotal * p.PageSize, p.CommitLimit * p.PageSize


class GpuCounters:
    """The process's GPU Process Memory counters (shared, dedicated, total committed), summed over its adapters.
    English counter names (PdhAddEnglishCounterW), so a localized Windows works too."""

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
            self.close()

    def close(self) -> None:
        if self.query:
            pdh.PdhCloseQuery(self.query)
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
                           if items[i].szName and items[i].szName.startswith(self.prefix)
                           and items[i].FmtValue.CStatus in (0, 1)))  # PDH_CSTATUS_VALID_DATA, NEW_DATA
        return out[0], out[1], out[2]


def census(h: int, pid: int, guest: GuestMap, every_region: bool = False):
    """Bytes per zone [resident, resident not shareable, committed, committed not resident]; each allocation outside
    the guest's as base: (zone, reserved, committed, resident, protection); ms the allocations, the working set and the
    longest single call took. Raises OSError when the process exits meanwhile, so no half census is written."""
    timer = Timer()
    t0 = time.perf_counter()
    allocs = allocations(h, timer)
    t1 = time.perf_counter()
    pages, shared, readonly = working_set(h)
    t2 = time.perf_counter()
    bases = [a[0] for a in allocs]

    def alloc_of(addr: int) -> int | None:
        i = bisect.bisect_right(bases, addr) - 1
        return bases[i] if i >= 0 and addr < bases[i] + allocs[i][1] else None

    block = guest_block(allocs)
    edges = guest.edges(block)
    heaps = heap_list(h)
    if heaps is None and exited(h):
        raise OSError("the process exited")
    owners = heap_owners(h, heaps, alloc_of) if heaps else {}
    stacks = stack_bases(h, pid)
    zones: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    per: dict[int, tuple[str, int, int, int, int]] = {}
    name = C.create_unicode_buffer(1024)

    def add(z: str, s: int, e: int, committed: int | None = None) -> int:
        """Pages [s, e) into zone z: resident from the working set, committed given (else all of [s, e))."""
        res = resident_in(pages, s, e)
        v = zones[z]
        v[0] += res
        v[1] += res - resident_in(shared, s, e)
        c = e - s if committed is None else committed
        v[2] += c
        v[3] += max(0, c - res)
        return res

    for base, reserved, committed, prot, typ in allocs:
        end = base + reserved
        if is_guest(base, reserved, block):
            if every_region:  # exact commit per zone, at the cost the docstring gives
                for rb, rs, state, rp in regions_of(h, base, reserved, timer):
                    if state == MEM_COMMIT:
                        for s, e in cut(rb, rb + rs, edges):
                            add(protection_part(guest.zone(s, base, reserved, block), rp), s, e)
                continue
            res_all = 0
            for s, e in cut(base, end, edges):  # resident by zone, read-only apart, from the working set alone
                z = guest.zone(s, base, reserved, block)
                ro = resident_in(readonly, s, e)
                rw = resident_in(pages, s, e) - ro
                for zz, n in ((z, rw), (z + " read-only", ro)):
                    if n:
                        v = zones[zz]
                        v[0] += n
                        v[1] += n
                        v[2] += n
                res_all += rw + ro
            if committed > res_all:
                v = zones[allocation_zone(base, reserved, block) + " not resident"]
                v[2] += committed - res_all
                v[3] += committed - res_all
            continue
        if typ in (MEM_IMAGE, MEM_MAPPED):
            got = psapi.GetMappedFileNameW(h, base, name, len(name))
            z = image_zone(name.value if got else "")
        else:
            guarded = False
            if base not in stacks and base not in owners and committed <= WALK_CAP and not prot & 0x600:
                guarded = any(state == MEM_COMMIT and rp & PAGE_GUARD
                              for _, _, state, rp in regions_of(h, base, reserved, timer))
            z = private_zone(reserved, committed, prot, guarded, base in stacks, owners.get(base))
        res = add(z, base, end, committed)
        per[base] = (z, reserved, committed, res, prot)
    if exited(h):
        raise OSError("the process exited")
    # Said only of a census that is written: a process that is ending loses its heaps before it is signalled (23/11/2026,
    # a run's last census warned this and was then dropped as the process exited)
    if heaps is None:
        warn("the PEB's heap list cannot be read: no malloc and heap zones")
    return zones, per, (round((t1 - t0) * 1000), round((t2 - t1) * 1000), round(timer.worst, 1))


def watch(out: Path, log: Path | None, every: float, detail: str | None, pid: int | None, phase_of,
          every_region: bool = False) -> None:
    """memwatch.py's loop on Windows: the same files and columns, with the meanings of this module's docstring."""
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS: the game's threads first
    if detail and detail.startswith(("guest ", "rt ")):
        warn(f"--detail {detail!r}: on Windows it lists allocations outside the guest's, e.g. 'gpu write-combined', "
             "'private unattributed 256 MiB-1 GiB', 'malloc'")
    while pid is None:
        pid = find_game()
        if pid is None:
            time.sleep(0.5)
    print(f"memwatch: watching mkw.exe PID {pid}", file=sys.stderr, flush=True)
    h = open_process(pid)
    gpu, guest, state, t0 = GpuCounters(pid), GuestMap(), {}, time.monotonic()
    det = out.with_suffix(".detail.tsv").open("w", encoding="utf-8", errors="replace") if detail else None
    if det:  # memwatch.py's columns: vmflags is the allocation's protection, below stays empty
        det.write("t\tstart\tsize\trss\tswap\tvmflags\tbelow\n")
    mib = lambda b: round(b / MiB)  # noqa: E731
    first = True
    with out.open("w", encoding="utf-8", errors="replace") as f:
        f.write("# os=win32: rss resident, pss resident not shareable, anon committed, swap committed not resident\n")
        f.write("t\tphase\tzone\trss\tpss\tanon\tswap\n")
        while not exited(h):
            if first and log and not log.is_file():
                warn(f"the log {log} is not there yet: the guest's zones follow memory.c's layout until it is")
            guest.read(log)
            try:
                zones, per, (alloc_ms, ws_ms, worst_ms) = census(h, pid, guest, every_region)
                ws, private_ws, private, peak = process_counters(h)
            except OSError:
                break
            try:
                phase = phase_of(log, state)
            except OSError:
                phase = state.get("phase", "")
            t = round(time.monotonic() - t0, 1)
            for z, v in sorted(zones.items()):
                if v[0] or v[2]:
                    f.write(f"{t}\t{phase}\t{z}\t" + "\t".join(str(mib(x)) for x in v) + "\n")
            avail, commit, limit = system_counters()
            g = gpu.read()
            if g is None:
                warn("the GPU Process Memory counters cannot be read: =gpu written as 0")
            g = g or (0, 0, 0)
            f.write(f"{t}\t{phase}\t=process\t{mib(ws)}\t{mib(private_ws)}\t{mib(private)}\t{mib(peak)}\n")
            f.write(f"{t}\t{phase}\t=system\t{mib(avail)}\t0\t{mib(commit)}\t{mib(limit)}\n")
            f.write(f"{t}\t{phase}\t=gpu\t{mib(g[0])}\t{mib(g[1])}\t{mib(g[2])}\t0\n")
            f.write(f"{t}\t{phase}\t=memwatch\t{alloc_ms}\t{ws_ms}\t{round(worst_ms)}\t0\n")  # whole ms, as MiB
            f.flush()
            if det:
                mine = sorted(((b, *v[1:]) for b, v in per.items() if v[0] == detail), key=lambda r: -r[2])
                if first and not mine:
                    warn(f"--detail {detail!r}: no allocation in that zone in the first census")
                for base, reserved, committed, res, prot in mine[:12]:
                    det.write(f"{t}\t{base:#x}\t{reserved / MiB:.1f}\t{res / MiB:.1f}\t"
                              f"{max(0, committed - res) / MiB:.1f}\t{prot:#x}\t\n")
                det.flush()
            first = False
            time.sleep(every)
    gpu.close()
    k32.CloseHandle(h)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--regions"]
    if sys.platform != "win32" or len(args) != 1 or not args[0].isdigit():
        sys.exit("usage (Windows): uv run tools/run/memwatch_win.py PID [--regions]   (memwatch.py OUT.tsv: a series)")
    pid = int(args[0])
    h = open_process(pid)
    zones, _, (alloc_ms, ws_ms, worst_ms) = census(h, pid, GuestMap(), "--regions" in sys.argv)
    ws, private_ws, private, peak = process_counters(h)
    gpu = GpuCounters(pid)
    g = gpu.read()
    gpu.close()
    print(f"{'zone':36} {'resident':>9} {'private':>8} {'commit':>8} {'not res':>8}   (MiB)")
    for z, v in sorted(zones.items(), key=lambda kv: -kv[1][2]):
        if v[2] >= MiB or v[0] >= MiB:
            print(f"{z:36} " + " ".join(f"{x / MiB:8,.0f}" for x in v))
    print(f"process: working set {ws / MiB:,.0f} (private {private_ws / MiB:,.0f}, peak {peak / MiB:,.0f}), private "
          f"commit {private / MiB:,.0f} MiB; census {alloc_ms} + {ws_ms} ms, longest call {worst_ms} ms")
    if g:
        print(f"gpu: shared {g[0] / MiB:,.0f}, dedicated {g[1] / MiB:,.0f}, committed {g[2] / MiB:,.0f} MiB")


if __name__ == "__main__":
    main()
