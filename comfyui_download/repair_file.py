#!/usr/bin/env python
"""Repair zeroed/corrupt regions in a downloaded safetensors file.

Scans every tensor fully for all-zero content, then re-downloads the exact
bytes for damaged tensors from the source URL and writes them in place.
"""
import struct, json, sys, urllib.request

def main(path, url):
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

    # Collect tensors to check (skip empty)
    tensors = [(k, m['data_offsets'][0], m['data_offsets'][1] - m['data_offsets'][0])
               for k, m in hdr.items() if isinstance(m, dict) and 'data_offsets' in m and m['data_offsets'][1] > m['data_offsets'][0]]
    print(f"{len(tensors)} tensors to check")

    damaged = []
    with open(path, 'rb') as f:
        for k, off, size in tensors:
            f.seek(data_start + off)
            data = f.read(size)
            if size >= 512:
                # fully-zero OR partially: use mean of abs as heuristic (corruption = zeros)
                nz = sum(1 for x in data if x != 0)
                if nz == 0:
                    damaged.append((k, off, size, 'ALL_ZERO'))
                elif nz < size * 0.01:  # >99% zero = corrupt
                    damaged.append((k, off, size, f'{100*nz//size}%_nonzero'))
    print(f"damaged tensors: {len(damaged)}")

    if not damaged:
        print("No damage found.")
        return 0

    # Repair each damaged tensor via range download from the source
    total = sum(sz for _, _, sz, _ in damaged)
    print(f"total repair bytes: {total/1e6:.1f} MB")
    repaired = 0
    with open(path, 'r+b') as f:
        for k, off, size, tag in damaged:
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={data_start+off}-{data_start+off+size-1}'})
                data = urllib.request.urlopen(req, timeout=120).read()
                if len(data) != size:
                    print(f"  [FAIL] {k[:50]}: got {len(data)} != {size}")
                    continue
                f.seek(data_start + off)
                f.write(data)
                repaired += 1
                if repaired % 50 == 0:
                    print(f"  repaired {repaired}/{len(damaged)}")
            except Exception as e:
                print(f"  [ERR] {k[:50]}: {str(e)[:60]}")
    print(f"repaired {repaired}/{len(damaged)}")
    return 0 if repaired == len(damaged) else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
