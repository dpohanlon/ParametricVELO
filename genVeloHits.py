import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl

mpl.use("Agg")

import seaborn

import time

from pprint import pprint

from scipy.optimize import linear_sum_assignment

import json

import time

plt.style.use(["seaborn-whitegrid", "seaborn-ticks"])
rcParams["figure.figsize"] = 12, 8
rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["figure.facecolor"] = "FFFFFF"

rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams["mathtext.fontset"] = "cm"
rcParams["mathtext.rm"] = "serif"

rcParams.update({"figure.autolayout": True})

import numpy as np

np.random.seed(42)

# 3 is not forward compatible with 4 (or 'unspecified'), but 'unspecified'
# is /sometimes/ 3 in /some/ LCG releases which makes this a bit of a
# nightmare to determine which to use. Hopefully this either works or breaks loudly.

try:
    import uproot3
except:
    import uproot

from scipy.stats import rv_histogram, expon, poisson

from veloGeom import buildTileXY, boundingVolumes, testIntersection

from genPhSp import genPhaseSpace

# I stole these from https://github.com/gcowan/RapidSim/tree/master/rootfiles/fonll,
# as the FONLL website wasn't working

# FONLL b(B?) dsigma/deta @ 14 TeV as numpy histogram
fonllb_eta = uproot.open("LHCb14.root")["eta"].to_numpy()
fonllb_pt = uproot.open("LHCb14.root")["pT"].to_numpy()

# FONLL c dsigma/deta @ 14 TeV as numpy histogram
fonllc_eta = uproot.open("LHCc14.root")["eta"].to_numpy()
fonllc_pt = uproot.open("LHCc14.root")["pT"].to_numpy()


def dist(h1, h2):
    return np.linalg.norm(h1 - h2)


def multipleScatterWidth(detConfig, p, thetas):

    siRadLength = 93  # mm
    siZ = 14
    planeThickness = detConfig["planeThickness"]

    velocity = 1.0  # beta c

    dx = planeThickness / np.cos(thetas[0])
    dy = planeThickness / np.cos(thetas[1])

    x = np.sqrt(dx ** 2 + dy ** 2)

    theta = (0.0136 / (velocity * p)) * siZ * (x / siRadLength)
    theta *= 1 + 0.038 * np.log(x / siRadLength)

    return theta


def calculateTrackPlaneThetas(ray):

    thetaX = np.arctan2(ray[0], ray[2])
    thetaY = np.arctan2(ray[1], ray[2])

    return thetaX, thetaY


def exchangeTrackHits(recoHits, frac=0.2, prob=0.2):

    newHits = recoHits.copy()

    # Exchange closest frac hits with probability prob
    # Only exchange hits on the same plane

    # Want to avoid exchanging hits for same tracks?

    for plane in range(N):

        planeHits = newHits[:, plane, :]

        k = int(len(planeHits) * frac)
        select = int(len(planeHits) * frac * prob)

        # Do the naive thing first, revisit if it's too slow

        dists = np.array(
            [
                [
                    (dist(h1, h2) if not np.all(np.equal(h1, h2)) else np.inf)
                    for h1 in planeHits
                ]
                for h2 in planeHits
            ]
        )

        # Choose a minimal distance assignment that corresponds to the swaps

        s = linear_sum_assignment(dists)
        s = list(zip(s[0], s[1]))

        # Pick the k nearest hits

        s = sorted(s, key=lambda p: dists[p])
        s = np.array(s[:k])

        # Choose select at random

        sIdx = np.random.choice(range(len(s)), select, replace=False)
        s = list(map(tuple, s[sIdx]))

        # Do the exchange

        planeHits[[x[0] for x in s], :], planeHits[[x[1] for x in s], :] = (
            planeHits[[x[1] for x in s], :],
            planeHits[[x[0] for x in s], :],
        )

    return newHits


def genTracks(geom, n, allFONLL=False):

    oXY = geom["tile"]["short"] + geom["beamGap"] + geom["shortT"]

    # origins = np.random.multivariate_normal(
    #     [oXY, oXY - 4, 0], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], n
    # )

    origins = np.array([[oXY, oXY - 4, 0] for i in range(n)])

    # Sperically uniform (in theta, phi)
    directions = np.random.uniform([0, 0], [np.pi, 2 * np.pi], [n, 2])

    etas, pts = sampleBKinematics(
        fonllb_eta, fonllb_pt, n
    )  # if allFONLL else directions[:,0]
    thetas = 2 * np.arctan(np.exp(-etas))

    # If not FONLL...
    # r = 1
    # directionsX = r * np.sin(thetas) * np.cos(directions[:,1])
    # directionsY = r * np.sin(thetas) * np.sin(directions[:,1])
    # directionsZ = r * np.cos(thetas)

    # px, py, pz
    directionsX = pts * np.cos(directions[:, 1])
    directionsY = pts * np.sin(directions[:, 1])
    directionsZ = pts * np.sinh(etas)

    directionsCart = np.stack((directionsX, directionsY, directionsZ), axis=1)

    s = np.sum(directionsCart ** 2, 1).reshape(-1, 1)

    return [(origins[i], directionsCart[i]) for i in range(len(directions))]


def sampleBKinematics(fonll_eta, fonll_pt, n):

    distEtas = rv_histogram(fonll_eta)
    etas = distEtas.rvs(size=n)

    distPts = rv_histogram(fonll_pt)
    pts = distPts.rvs(size=n)

    return etas, pts


def formDecayproducts(
    flightDist, parentMass, parentBoost, productMass, nDecayTracks, track
):

    rayO, rayD = track

    pSq = rayD[0] ** 2 + rayD[1] ** 2 + rayD[2] ** 2

    # Distance travelled per ray param, t
    dPerT = np.sqrt(pSq)
    flightDist = flightDist.rvs(size=1).squeeze()

    # t for decay origin
    t = flightDist / dPerT

    decayOrigin = (rayO[0] + rayD[0] * t, rayO[1] + rayD[1] * t, rayO[2] + rayD[2] * t)

    # Calculate momentum (direction) of decay tracks according to phase-space
    # distribution of N tracks. Boost from parent rest frame using parent momentum

    parentE = np.sqrt(pSq + parentMass ** 2) + np.random.normal(
        parentBoost[0], parentBoost[1]
    )
    decayMasses = [productMass for i in range(nDecayTracks)]

    decayProdP = genPhaseSpace(
        decayMasses, parentMass, np.array(list(rayD) + [parentE])
    )

    decayTracks = []
    for t in decayProdP:
        decayTracks.append((decayOrigin, list(t[:-1])))

    return decayTracks


def addDecays(decayParams, tracks, avgBB=1):

    # Let's not worry about cascade decays, or double D decays for the moment

    bFlightDist = expon(scale=decayParams["Bflight"])
    dFlightDist = expon(scale=decayParams["Dflight"])

    nB = np.random.poisson(avgBB) * 2  # Always a pair
    nD = np.random.poisson(avgBB * decayParams["DtoBProd"])

    bTracks = tracks[:nB]
    dTracks = tracks[nB : nB + nD]

    nTracksFromB = np.random.choice(
        range(2, len(decayParams["BtoN"]) + 2), p=decayParams["BtoN"], size=nB
    )
    nTracksFromD = np.random.choice(
        range(2, len(decayParams["DtoN"]) + 2), p=decayParams["DtoN"], size=nD
    )

    decayTracks = []
    for iTrack, track in enumerate(bTracks):

        products = formDecayproducts(
            bFlightDist,
            decayParams["BMass"],
            (decayParams["EBoost"], decayParams["EBoostSigma"]),
            decayParams["piMass"],
            nTracksFromB[iTrack],
            track,
        )
        decayTracks += products

    for iTrack, track in enumerate(dTracks):

        products = formDecayproducts(
            dFlightDist,
            decayParams["DMass"],
            (decayParams["EBoost"], decayParams["EBoostSigma"]),
            decayParams["piMass"],
            nTracksFromD[iTrack],
            track,
        )
        decayTracks += products

    # Slice off the parent particles, and stick on the decay tracks
    tracks = tracks[nB + nD :] + decayTracks

    return tracks
    # return decayTracks


def getTileHits(ray, tileInt, tileVol, tileName, pixelEdges):

    pixelEdgesShort, pixelEdgesLong = pixelEdges

    rayO, rayD = ray

    # xy of intersection using ray equation, assuming it hits the 'front'
    # (small z first) for the moment (idx 0)

    xHit = rayO[0] + rayD[0] * tileInt
    yHit = rayO[1] + rayD[1] * tileInt
    zHit = rayO[2] + rayD[2] * tileInt

    # Global space to tile space

    tileX0 = tileVol[0][0]
    tileY0 = tileVol[1][0]

    xHitTile = xHit - tileX0
    yHitTile = yHit - tileY0

    # Work out whether which of x/y is short/long

    edgesX, edgesY = (
        (pixelEdgesShort, pixelEdgesLong)
        if tileName in ["left", "right"]
        else (pixelEdgesLong, pixelEdgesShort)
    )

    xPix = np.digitize(xHitTile, edgesX, right=True)
    yPix = np.digitize(yHitTile, edgesY, right=True)

    xDigi = 0.5 * (edgesX[xPix - 1] + edgesX[xPix])
    yDigi = 0.5 * (edgesY[yPix - 1] + edgesY[yPix])

    # Back to global coords for plotting

    xDigi_global = tileX0 + xDigi
    yDigi_global = tileY0 + yDigi

    return (xDigi_global, yDigi_global), (xPix, yPix)


def genHits(nGen=10, tracks=None):

    geom = json.load(open("veloGeom.json", "r"))
    decayParams = json.load(open("decayProbs.json", "r"))

    hitDropProb = 1.0 - geom["hitEfficiency"]

    bottom, right, top, left = buildTileXY(geom)

    volsA, volsC = boundingVolumes(geom, (bottom, right, top, left))

    pixelEdgesShort = np.linspace(
        0, geom["tile"]["short"], geom["pixelsPerTile"]["short"] + 1
    )  # 28.16 mm, 512 pixels, 55 microns wide
    pixelEdgesLong = np.linspace(
        0, geom["tile"]["long"], geom["pixelsPerTile"]["long"] + 1
    )  # 42.24 mm, 768 pixels, 55 microns wide

    if tracks == None:

        # Generate tracks according to FONLL distribution (add min bias (spherical) later)
        tracks = genTracks(geom, nGen, allFONLL=True)

        # Select only those going forwards for now, *in principle* a cut on eta
        tracks = list(
            filter(
                lambda x: np.arcsinh(x[1][2] / np.sqrt(x[1][0] ** 2 + x[1][1] ** 2))
                > 3,
                tracks,
            )
        )

        # Replace selected tracks from the PV with decay tracks
        tracks = np.array(addDecays(decayParams, tracks))

    hits = []
    hitsPix = []

    tileIndex = {"bottom": 0, "right": 1, "top": 2, "left": 3}

    # Implement an ad-hoc bounding-volume hierarchy (i.e., test intersections first with
    # coarse bounding volumes rather than individual pixels).
    for iRay, ray in enumerate(tracks):

        trackHits = []
        trackHitsPix = []

        ms = np.array([0.0, 0.0])
        trackP = np.sqrt(np.sum(np.array(ray[1]) ** 2))  # GeV
        trackThetas = calculateTrackPlaneThetas(ray[1])

        msWidth = multipleScatterWidth(geom, trackP, trackThetas)  # mm
        # msDist -> np.random.normal(0, msWidth), for x and y independently

        for cSide, vols in zip([True, False], [volsC, volsA]):

            # Instead of matching pixels in a square cell, match them in 3 x 2 cell tiles instead
            for ibv, bv in enumerate(vols):

                # t of intersection, ray = o + d * t
                bvInt = testIntersection(ray, bv)

                # If there's an ineteraction with a bounding volume
                if bvInt != False:

                    z = bv[2]
                    tileVols = (
                        np.array([[t[0], t[1], z] for t in [bottom, left]])
                        if cSide
                        else np.array([[t[0], t[1], z] for t in [right, top]])
                    )
                    names = ["bottom", "left"] if cSide else ["right", "top"]

                    # Test which tile in the station it intersected (could be none!)
                    for tileName, tileVol in zip(names, tileVols):

                        tileInt = testIntersection(ray, tileVol)
                        dropHit = np.random.choice(
                            [True, False], p=[hitDropProb, 1 - hitDropProb]
                        )

                        # If there's an interaction with a tile
                        if tileInt != False and dropHit != True:

                            (xDigi_global, yDigi_global), (xPix, yPix) = getTileHits(
                                ray,
                                tileInt,
                                tileVol,
                                tileName,
                                (pixelEdgesShort, pixelEdgesLong),
                            )

                            msUpdate = np.random.normal(0, msWidth, size=2)
                            ms += msUpdate

                            rayWithMS = (
                                [ray[0][0] + ms[0], ray[0][1] + ms[1], ray[0][2]],
                                ray[1],
                            )

                            # Update tileInt for ray with MS perturbation
                            tileInt = testIntersection(rayWithMS, tileVol)

                            # Bail if this actually takes us out of acceptance
                            if tileInt == False:
                                ms -= msUpdate
                                continue

                            (xDigi_global, yDigi_global), (xPix, yPix) = getTileHits(
                                rayWithMS,
                                tileInt,
                                tileVol,
                                tileName,
                                (pixelEdgesShort, pixelEdgesLong),
                            )

                            trackHits.append(
                                (xDigi_global, yDigi_global, z[0] + (z[1] - z[0]) / 2.0)
                            )
                            trackHitsPix.append(
                                (ibv, tileIndex[tileName], xPix, yPix)
                            )  # z index, tileIndex, pixelX, pixelY

                        elif dropHit == True:

                            trackHits.append(
                                (np.nan, np.nan, z[0] + (z[1] - z[0]) / 2.0)
                            )
                            trackHitsPix.append(
                                (ibv, tileIndex[tileName], np.nan, np.nan)
                            )  # z index, tileIndex, pixelX, pixelY

        # hits.append(trackHits)
        # hitsPix.append(trackHitsPix)

        # Pad hits to 26 (nStations)

        hPad = np.zeros((26, 3)) * np.nan
        hPixPad = np.zeros((26, 4)) * np.nan

        hPad[: len(trackHits), :] = np.array(trackHits).reshape(-1, 3)
        hPixPad[: len(trackHitsPix), :] = np.array(trackHitsPix).reshape(-1, 4)

        hits.append(hPad)
        hitsPix.append(hPixPad)

    hits = np.array(hits)
    hitsPix = np.array(hitsPix)

    # Pad tracks to 250

    tracksPad = np.zeros((250, 2, 3)) * np.nan
    hitsPad = np.zeros((250, 26, 3)) * np.nan
    hitsPixPad = np.zeros((250, 26, 4)) * np.nan

    tracksPad[: len(tracks), :] = np.array(tracks)
    hitsPad[: len(hits), :] = hits
    hitsPixPad[: len(hitsPix), :] = hitsPix

    return tracksPad, hitsPad, hitsPixPad


def genEventParallel(lock, i):

    np.random.seed(i)

    tracksArray = []
    hitsArray = []
    hitsPixArray = []

    n = 1000

    for e in range(n):

        tracks, hits, hitsPix = genHits(nGen=350)
        tracksArray.append(tracks)
        hitsArray.append(hits)
        hitsPixArray.append(hitsPix)

    from time import sleep

    print(i, "waiting")
    lock.acquire()
    print(i, "acquired")

    try:

        file = h5py.File("/data/1/dan/veloData_ms.h5", "a")

        file["tracks"].resize((file["tracks"].shape[0] + n), axis=0)
        file["tracks"][-n:] = np.array(tracksArray)

        file["hits"].resize((file["hits"].shape[0] + n), axis=0)
        file["hits"][-n:] = np.array(hitsArray)

        file["hitsPix"].resize((file["hitsPix"].shape[0] + n), axis=0)
        file["hitsPix"][-n:] = np.array(hitsPixArray)

        file.close()

    finally:
        lock.release()
        print(i, "released")


from threading import Lock, Semaphore
import h5py

# if __name__ == '__main__':
#
#     pprint(genHits(nGen=250))

if __name__ == "__main__":

    from tqdm import tqdm

    from functools import partial

    from multiprocessing import Pool, Manager

    p = Pool(16)

    n = 1000

    tracks, hits, hitsPix = genHits(nGen=350)

    file = h5py.File("/data/1/dan/veloData_ms.h5", "w")

    file.create_dataset(
        "tracks",
        data=tracks.reshape(1, 250, 2, 3),
        maxshape=(None, 250, 2, 3),
        compression="lzf",
    )
    file.create_dataset(
        "hits",
        data=hits.reshape(1, 250, 26, 3),
        maxshape=(None, 250, 26, 3),
        compression="lzf",
    )
    file.create_dataset(
        "hitsPix",
        data=hitsPix.reshape(1, 250, 26, 4),
        maxshape=(None, 250, 26, 4),
        compression="lzf",
    )

    file.close()

    # Need a Manager because locks can't be pickled, and processes fork()

    manager = Manager()
    lock = manager.Lock()

    parallelFunc = partial(genEventParallel, lock)

    p.map(parallelFunc, range(n - 1))

    # for i in tqdm(range(n - 1)):
    #     tracks, hits, hitsPix = genHits(nGen=350)
    #
    #     file["tracks"].resize((file["tracks"].shape[0] + 1), axis = 0)
    #     file["tracks"][-1:] = tracks
    #
    #     file["hits"].resize((file["hits"].shape[0] + 1), axis = 0)
    #     file["hits"][-1:] = hits
    #
    #     file["hitsPix"].resize((file["hitsPix"].shape[0] + 1), axis = 0)
    #     file["hitsPix"][-1:] = hitsPix
