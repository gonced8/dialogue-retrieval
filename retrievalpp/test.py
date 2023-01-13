b = 5
for x in range(b):
    for y in range(b):
        if y == x:
            continue
        for z in range(y + 1, b):
            if z == x:
                continue
            print(y, z)
            input()
