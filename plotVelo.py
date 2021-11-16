import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl

mpl.use("Agg")

import seaborn

import time

from pprint import pprint

import json

import numpy as np

np.random.seed(42)

import uproot

from scipy.stats import rv_histogram, expon, poisson

from veloGeom import buildTileXY, boundingVolumes, testIntersection

from genPhSp import genPhaseSpace

from genVeloHits import genTracks, addDecays
from genVeloHits import formDecayproducts, sampleBKinematics, genHits

plt.style.use(["seaborn-whitegrid", "seaborn-ticks"])
rcParams["figure.figsize"] = 12, 12
rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["figure.facecolor"] = "FFFFFF"

rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams["mathtext.fontset"] = "cm"
rcParams["mathtext.rm"] = "serif"

rcParams.update({"figure.autolayout": True})


def drawXY(tile, colour="k"):

    # bottom
    plt.plot([tile[0][0], tile[0][1]], [tile[1][0], tile[1][0]], color=colour)

    # right
    plt.plot([tile[0][1], tile[0][1]], [tile[1][0], tile[1][1]], color=colour)

    # top
    plt.plot([tile[0][0], tile[0][1]], [tile[1][1], tile[1][1]], color=colour)

    # left
    plt.plot([tile[0][0], tile[0][0]], [tile[1][0], tile[1][1]], color=colour)


def drawZ(geom, xRangeA, xRangeC):

    for z in geom["z"]["a"]:
        plt.plot([z, z], [xRangeA[0], xRangeA[1]], color="red")

    for z in geom["z"]["c"]:
        plt.plot([z, z], [xRangeC[0], xRangeC[1]], color="black")


def drawXYTrack(geom, ray):

    rayO, rayD = ray

    xMax = 500

    ts = np.linspace(0, 1000, 100)

    # z-x projection

    pointsX = [rayO[0] + rayD[0] * t for t in ts]
    pointsY = [rayO[1] + rayD[1] * t for t in ts]

    plt.plot(pointsX, pointsY, linestyle=":", alpha=0.5)


def drawTrack(geom, ray):

    rayO, rayD = ray

    zMax = geom["z"]["a"][-1] + 10.0

    maxT = (zMax - rayO[2]) / rayD[2]
    ts = np.linspace(0, 1000, 100)

    # z-x projection

    pointsX = [rayO[0] + rayD[0] * t for t in ts]
    pointsZ = [rayO[2] + rayD[2] * t for t in ts]

    plt.plot(pointsZ, pointsX, linestyle=":", alpha=0.5)


if __name__ == "__main__":

    geom = json.load(open("veloGeom.json", "r"))
    decayParams = json.load(open("decayProbs.json", "r"))

    bottom, right, top, left = buildTileXY(geom)  # , offset = (10, 10))

    drawXY(bottom, "black")
    drawXY(right, "blue")
    drawXY(top, "red")
    drawXY(left, "green")

    plt.savefig("test.pdf")
    plt.clf()

    volsA, volsC = boundingVolumes(geom, (bottom, right, top, left))

    xRangeA = (top[0][0], top[0][1])
    xRangeC = (bottom[0][0], bottom[0][1])

    drawZ(geom, xRangeA, xRangeC)

    plt.xlim(-400, 800)
    plt.ylim(-10, 100)

    plt.savefig("testZ.pdf")
    plt.clf()

    tracks = genTracks(geom, 1000, allFONLL=True)

    # Select only those going forwards for now, *in principle* a cut on eta
    tracks = list(
        filter(
            lambda x: np.arcsinh(x[1][2] / np.sqrt(x[1][0] ** 2 + x[1][1] ** 2)) > 3,
            tracks,
        )
    )
    print("N tracks:", len(tracks))

    tracks = addDecays(decayParams, tracks)

    drawXY(bottom, "black")
    drawXY(right, "blue")
    drawXY(top, "red")
    drawXY(left, "green")

    tracks, hits, hitsPix = genHits(nGen=1000, tracks=tracks)

    for t in hits:
        plt.plot([h[0] for h in t], [h[1] for h in t], "+")

    plt.xlim(-10, 80)
    plt.ylim(-10, 80)

    plt.savefig("hitXY.pdf")
    plt.clf()

    drawXY(bottom, "black")
    drawXY(right, "blue")
    drawXY(top, "red")
    drawXY(left, "green")

    for r in np.array(tracks):  # [list(toDraw)]:
        drawXYTrack(geom, r)

    plt.xlim(-10, 80)
    plt.ylim(-10, 80)

    plt.savefig("trackXY.pdf")
    plt.clf()

    drawZ(geom, xRangeA, xRangeC)

    for track in np.array(tracks):  # [list(toDraw)]:
        drawTrack(geom, track)

    plt.xlim(-400, 800)
    plt.ylim(-10, 90)

    plt.savefig("trackZ.pdf")

    plt.xlim(-50, 50)
    plt.ylim(35, 40)

    plt.savefig("trackZZoom.pdf")

    # emptyTile = np.zeros((geom["pixelsPerTile"]["short"], geom["pixelsPerTile"]["long"]))
    # for i in range(10000):
    #     rx = np.random.randint(0, geom["pixelsPerTile"]["short"])
    #     ry = np.random.randint(0, geom["pixelsPerTile"]["long"])
    #     emptyTile[rx][ry] = 1
    #
    # plt.imshow(emptyTile)
    # plt.savefig('tileTest.pdf')
