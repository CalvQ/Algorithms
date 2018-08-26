w1 = [1957, 2431, 3495, 3857, 4127, 4439, 4467, 5754, 9128, 9408, 9949]
w2 = [790, 1228, 1468, 1655, 1854, 2412, 2506, 2883, 5060, 5539, 6061]
w3 = [1291, 4415, 5536, 5978, 6062, 6983, 8448, 8473]

(start, end) = (min(w1 + w2 + w3), max(w1 + w2 + w3))

t = {}

for i in range(min(w1 + w2 + w3), max(w1 + w2 + w3) + 1):
    if i in w1:
        t['w1'] = i

    if i in w2:
        t['w2'] = i

    if i in w3:
        t['w3'] = i

    if 'w1' in t and 'w2' in t and 'w3' in t:
        if max(t.values()) - min(t.values()) < end - start:
            (start, end) = (min(t.values()), max(t.values()))

        if t['w1'] == start:
            del t['w1']

        if t['w2'] == start:
            del t['w2']

        if t['w3'] == start:
            del t['w3']

print([start, end])
