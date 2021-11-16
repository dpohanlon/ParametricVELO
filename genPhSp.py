import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl

mpl.use("Agg")

# ROOT calls this the PDK function, which seems as good a name as any
def pdk(a, b, c):

    x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)

    return np.sqrt(x) / (2 * a)


def boost(fourVec, boostVec):

    boostVec = np.array(boostVec)

    bSq = np.sum(boostVec ** 2)

    gamma = 1.0 / np.sqrt(1 - bSq)

    bP = np.sum(boostVec * fourVec[:-1])

    gammaSq = (gamma - 1.0) / bSq if bSq > 0 else 0.0

    e = fourVec[-1]

    # TODO: Make this a matrix boost

    px = fourVec[0] + gammaSq * bP * boostVec[0] + gamma * boostVec[0] * e
    py = fourVec[1] + gammaSq * bP * boostVec[1] + gamma * boostVec[1] * e
    pz = fourVec[2] + gammaSq * bP * boostVec[2] + gamma * boostVec[2] * e

    return [px, py, pz, gamma * (e + bP)]


# Basically a 1 to 1 copy of the TGenPhaseSpace implementation of Raubold-Lynch-style sampling
# (https://root.cern/doc/master/TGenPhaseSpace_8cxx_source.html)
def genPhaseSpace(masses, parentMass, parent4v):

    n = len(masses)
    products4v = np.zeros((n, 4))

    mag3v = np.sqrt(np.sum(parent4v[:-1] ** 2))
    mag4v = np.sqrt(parent4v[-1] ** 2 - mag3v ** 2)

    parentBeta = mag4v / parent4v[-1]
    betaWeight = parentBeta / mag3v if abs(mag3v) > 1e-8 else 0.0
    frameBeta = [
        parent4v[0] * betaWeight,
        parent4v[1] * betaWeight,
        parent4v[2] * betaWeight,
    ]

    eMinusMass = mag4v - np.sum(masses)

    # In principle there should also be some (de-biasing?) weights, but who cares
    rand = np.sort(np.random.uniform(size=n))
    rand[0] = 0
    rand[-1] = 1

    cs = np.cumsum(masses)

    # Masses of intermediate decay products
    intMasses = eMinusMass * rand + cs

    pd = [pdk(intMasses[i + 1], intMasses[i], masses[i + 1]) for i in range(0, n - 1)]

    e0 = np.sqrt(pd[0] ** 2 + masses[0] ** 2)

    products4v[0, :] = [0, pd[0], 0, e0]

    # TO DO: Fix this w e i r d loop
    i = 1
    while True:

        ei = np.sqrt(pd[i - 1] ** 2 + masses[i] ** 2)
        products4v[i, :] = [0, -pd[i - 1], 0, ei]

        cZ = 2 * np.random.uniform() - 1
        sZ = np.sqrt(1 - cZ ** 2)

        angY = 2 * np.pi * np.random.uniform()

        cY = np.cos(angY)
        sY = np.sin(angY)

        for j in range(i + 1):

            pxj = products4v[j][0].copy()
            pyj = products4v[j][1].copy()

            # Rotate about Z
            products4v[j][0] = cZ * pxj - sZ * pyj
            products4v[j][1] = sZ * pxj + cZ * pyj

            # Rotate about Y
            pxj = products4v[j][0].copy()  # The new one!
            pzj = products4v[j][2].copy()

            products4v[j][0] = cY * pxj - sY * pzj
            products4v[j][2] = sY * pxj + cY * pzj

        if i == n - 1:
            break

        beta = pd[i] / np.sqrt(pd[i] ** 2 + intMasses[i] ** 2)

        for j in range(i + 1):
            products4v[j, :] = boost(products4v[j, :], [0, beta, 0])
        i += 1

    for i in range(n):
        products4v[i, :] = boost(products4v[i, :], frameBeta)

    return products4v


if __name__ == "__main__":

    masses = np.array([0.49, 0.49, 0.49])
    parentMass = 5.3
    parent4v = np.array([0, 0, 0, parentMass])

    m13Sq = []
    m23Sq = []

    for i in range(10000):

        fourVecs = genPhaseSpace(masses, parentMass, parent4v)

        p13 = fourVecs[0, :] + fourVecs[2, :]
        p23 = fourVecs[1, :] + fourVecs[2, :]

        mag3v2_13 = np.sum(p13[:-1] ** 2)
        mag4v2_13 = p13[-1] ** 2 - mag3v2_13

        mag3v2_23 = np.sum(p23[:-1] ** 2)
        mag4v2_23 = p23[-1] ** 2 - mag3v2_23

        m13Sq.append(mag4v2_13)
        m23Sq.append(mag4v2_23)

    # LGTM
    plt.plot(m13Sq, m23Sq, ".", markersize=1.0)
    plt.savefig("dpTest.pdf")
