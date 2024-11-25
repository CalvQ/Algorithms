# https://archive.topcoder.com/ProblemStatement/pm/17946

def paperWrap(paperArea):
    remainingPaper = paperArea
    totalVolume = 0

    while remainingPaper > 5:
        bestBox = None
        maxVolume = 0

        # Find best box
        for A in range(1, remainingPaper // 6 + 1):
            for B in range(1, remainingPaper // (2*A) + 1):
                for C in range(1, remainingPaper // (2*(A+B)) + 1):
                    surface_area = 2*(A*B + B*C + C*A)
                    volume = A*B*C

                    if surface_area > remainingPaper:
                        break

                    if volume > maxVolume or (volume == maxVolume and A < bestBox[0]) or (volume == maxVolume and A == bestBox[0] and B < bestBox[1]):
                        bestBox = (A, B, C)
                        maxVolume = volume

        if bestBox is None:
            break

        A, B, C = bestBox
        # print("BOX DIMS: ", A, B, C)
        # print("SA", 2 * (A*B + B*C + C*A))
        # print("VOLUME:", A * B * C)
        remainingPaper -= 2 * (A*B + B*C + C*A)
        totalVolume += A * B * C

    return totalVolume


assert paperWrap(605) == 1000
assert paperWrap(366) == 451
assert paperWrap(887) == 1734
assert paperWrap(888) == 1728
assert paperWrap(24174) == 254088
