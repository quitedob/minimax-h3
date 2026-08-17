#!/usr/bin/env python
"""Integrity scan for a safetensors file: per-tensor all-zero / >99%-zero detection.

Same heuristic as repair_file.py but read-only (no repair). Exits 0 if clean,
1 if any tensor looks corrupt.
"""
import struct, json, sys

def main(path):
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

    tensors = [(k, m['data_offsets'][0], m['data_offsets'][1] - m['data_offsets'][0])
               for k, m in hdr.items() if isinstance(m, dict) and 'data_offsets' in m and m['data_offsets'][1] > m['data_offsets'][0]]
    print(f"{len(tensors)} tensors to check")

    bad = []
    total = 0
    with open(path, 'rb') as f:
        for i, (k, off, size) in enumerate(tensors):
            total += size
            if size < 512:
                continue
            f.seek(data_start + off)
            data = f.read(size)
            nz = sum(1 for x in data if x != 0)
            if nz == 0:
                bad.append((k, size, 'ALL_ZERO'))
            elif nz < size * 0.01:
                bad.append((k, size, f'{100*nz//size}%_nonzero'))
            if (i + 1) % 50 == 0:
                print(f"  scanned {i+1}/{len(tensors)} ({(total//1024**2)//1024} GiB) ...", flush=True)

    print(f"\nscanned {total/1e9:.2f} GB, damaged tensors: {len(bad)}")
    for k, sz, tag in bad[:20]:
        print(f"  [DAMAGED] {k[:60]} size={sz/1e6:.1f}MB {tag}")
    if bad:
        print("RESULT: CORRUPT")
        return 1
    print("RESULT: OK (no zeroed/corrupt tensors)")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
