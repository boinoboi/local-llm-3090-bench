import sys

lines = open(sys.argv[1]).read().split("\n")
bad = []
for i, l in enumerate(lines, 1):
    if l.strip().startswith("|"):
        bad.append((i, l.count("|")))

blocks, cur, prev = [], [], 0
for i, n in bad:
    if i != prev + 1 and cur:
        blocks.append(cur)
        cur = []
    cur.append((i, n))
    prev = i
if cur:
    blocks.append(cur)

issues = [b for b in blocks if len({n for _, n in b}) > 1]
print(f"{len(blocks)} table blocks, {len(issues)} with inconsistent column counts")
for b in issues[:6]:
    print("  lines", b[0][0], "-", b[-1][0], "counts", sorted({n for _, n in b}))
