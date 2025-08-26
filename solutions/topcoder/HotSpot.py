# https://archive.topcoder.com/ProblemStatement/pm/18003

def hotSpot(points):
    return sum(points)/len(points)


assert (hotSpot([1, 5]) == 3.0)
assert (hotSpot([5]) == 5.0)
assert (hotSpot([1, 10, 10]) == 7.0)
assert (hotSpot([4, 7]) == 5.5)
assert (hotSpot([99, 14, 62, 3]) == 44.5)
assert (hotSpot([1, 2, 4]) == 7/3)
assert (hotSpot([1, 2, 2, 2, 2]) == 1.8)
