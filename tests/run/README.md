# tests/run

The tools of `tools/run/` that watch a running game, checked without the game.

| File | What it checks |
|---|---|
| `test_memwatch_win.py` | `tools/run/memwatch_win.py`: the guest's zones from `memory.c`'s layout and the log's lines, private memory, images and the Windows report of `memwatch.py` on any platform; on Windows, a census of a child process that commits and touches known amounts and grows a heap (its allocation, its heap and the process heap, its thread's stack, the zones against the process's private commit), and `memwatch.py` following it until it exits |
