#!/usr/bin/env python
"""Repair a large contiguous zeroed region in a downloaded safetensors file.

Scans for all-zero 64KB blocks, merges into contiguous ranges, then
re-downloads those exact byte ranges from the source and writes them back.
"""
import struct, json, sys, numpy as np, mmap, urllib.request, time

def main(path, url):
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm[data_start:], dtype=np.uint8)

    bs = 65536
    zero_start = None
    ranges = []
    for i in range(0, len(data), bs):
        chunk = data[i:i+bs]
        if chunk.size > 0 and not np.any(chunk):
            if zero_start is None:
                zero_start = i
        else:
            if zero_start is not None:
                ranges.append((zero_start, i - zero_start))
                zero_start = None
    if zero_start is not None:
        ranges.append((zero_start, len(data) - zero_start))

    total = sum(s for _, s in ranges)
    print(f"zero ranges: {len(ranges)}, total {total/1e6:.1f} MB ({total/len(data)*100:.1f}%)")
    if total == 0:
        print("No zero ranges found.")
        return 0

    # Merge adjacent ranges
    merged = []
    for a, s in ranges:
        if merged and a <= merged[-1][0] + merged[-1][1] + bs:
            merged[-1] = (merged[-1][0], (a + s) - merged[-1][0])
        else:
            merged.append((a, s))
    print(f"merged ranges: {len(merged)}")

    # Download + write each range
    CHUNK = 16 * 1024 * 1024  # 16MB chunks
    done = 0
    for a, s in merged:
        abs_start = data_start + a
        written = 0
        with open(path, 'r+b') as f:
            while written < s:
                lo = abs_start + written
                hi = min(lo + CHUNK, abs_start + s) - 1
                for attempt in range(5):
                    try:
                        req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}'})
                        blk = urllib.request.urlopen(req, timeout=180).read()
                        if len(blk) != hi - lo + 1:
                            raise ValueError(f"short read {len(blk)}")
                        f.seek(lo)
                        f.write(blk)
                        break
                    except Exception as e:
                        if attempt == 4:
                            print(f"  [FAIL] range {lo}-{hi}: {e}")
                            return 1
                        time.sleep(2)
                written += hi - lo + 1
        done += s
        print(f"  range done: {(a)/1e6:.1f}MB +{s/1e6:.1f}MB, total {done/1e6:.1f}/{total/1e6:.1f}MB")
    print("repair complete")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
