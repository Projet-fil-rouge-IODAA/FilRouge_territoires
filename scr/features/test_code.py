import json

with open("data/pixels/lab_px_test_invert.txt") as f:
    data = f.read()

pixels_selec = json.loads(data)

pixels_selec_invert = pixels_selec

for k,v in pixels_selec.items():
    print(k,v)
    v2 = []
    for px in v:
        v2.append([px[1],px[0]])
    pixels_selec_invert[k] = v2

with open("data/pixels/lab_px_test.txt", "w") as fp:
    json.dump(pixels_selec_invert, fp)
print("File ready.")