import numpy as np


def buildBottomTile(geom, offset):

    xLow = 0
    yLow = 0

    xHigh = xLow + geom["tile"]["long"]
    yHigh = yLow + geom["tile"]["short"]

    return (xLow + offset[0], xHigh + offset[0]), (yLow + offset[1], yHigh + offset[1])


def buildRightTile(geom, offset):

    xLow = geom["tile"]["long"]
    yLow = -geom["shortT"]

    xHigh = xLow + geom["tile"]["short"]
    yHigh = yLow + geom["tile"]["long"]

    return (xLow + offset[0], xHigh + offset[0]), (yLow + offset[1], yHigh + offset[1])


def buildTopTile(geom, offset):

    xLow = geom["shortT"] + geom["tile"]["short"]
    yLow = -geom["shortT"] + geom["tile"]["long"]

    xHigh = xLow + geom["tile"]["long"]
    yHigh = yLow + geom["tile"]["short"]

    return (xLow + offset[0], xHigh + offset[0]), (yLow + offset[1], yHigh + offset[1])


def buildLeftTile(geom, offset):

    xLow = geom["shortT"]
    yLow = geom["tile"]["short"]

    xHigh = xLow + geom["tile"]["short"]
    yHigh = yLow + geom["tile"]["long"]

    return (xLow + offset[0], xHigh + offset[0]), (yLow + offset[1], yHigh + offset[1])


def buildTileXY(geom, offset=(0, 0)):

    bottom = buildBottomTile(geom, offset)
    right = buildRightTile(geom, offset)
    top = buildTopTile(geom, offset)
    left = buildLeftTile(geom, offset)

    return bottom, right, top, left


def boundingVolumes(geom, tiles):

    bottom, right, top, left = tiles

    xRangeA = (top[0][0], top[0][1])
    xRangeC = (bottom[0][0], bottom[0][1])

    yRangeA = (right[1][0], top[1][1])
    yRangeC = (bottom[1][0], left[1][1])

    volsA = np.array([[xRangeA, yRangeA, (z - 0.5, z + 0.5)] for z in geom["z"]["a"]])
    volsC = np.array([[xRangeC, yRangeC, (z - 0.5, z + 0.5)] for z in geom["z"]["c"]])

    return volsA, volsC


def testIntersection(ray, volume):

    # Assume volume is axis aligned
    # ray: ((origin_x, origin_y, origin_z), (direction_x, direction_y, direction_z))
    # volume ((x0, x1), (y0, y1), (z0, z1))

    # Min and max extent

    a = volume[:, 0]
    b = volume[:, 1]

    o, d = ray

    tA = (a - o) / (d + 1e-12)
    tB = (b - o) / (d + 1e-12)

    tmin = max(max(min(tA[0], tB[0]), min(tA[1], tB[1])), min(tA[2], tB[2]))
    tmax = min(min(max(tA[0], tB[0]), max(tA[1], tB[1])), max(tA[2], tB[2]))

    t = tmin

    if tmax < 0 or tmin > tmax:
        t = tmax
        return False

    return t


def testTestIntersection():

    volume = np.array(((0, 42.24), (0, 70.4), (112.0, 113.0)))
    ray = np.array([[10.0, 10.0, 0], [0.0, 0.0, 1.0]])

    print(testIntersection(ray, volume))


if __name__ == "__main__":
    testTestIntersection()
