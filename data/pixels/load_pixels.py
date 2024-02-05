import json

with open("data/pixels/lab_px_test.txt") as f:
    data = f.read()

pixels_selec = json.loads(data)
