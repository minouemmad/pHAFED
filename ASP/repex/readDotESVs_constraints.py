import os
import copy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math

# Returns an 2D where each array are the coordinates of a HH curve
def generateData(listOfN, listOfpH, pKa):
    result = []
    for n in range(len(listOfN)):
        result.append([])
        for pH in listOfpH:
            result[n].append(hendersonHasselBalch(pH, listOfN[n], pKa))
    return result


# Returns the result of putting x into the HH with hill=n and pKa=pKa
def hendersonHasselBalch(x, n, pKa):
    superScript = n * (pKa - x)
    y = 1 / (1 + 10.0 ** superScript)  # math.pow is not vectorized
    return y


# Returns a 2D array of each list summed down its tautomer state
# Expects a 3D array of histograms (including the header)
def sumAcrossTautomerStates(data):
    temp = []
    for i in range(len(data)):
        temp.append([])
        for j in range(len(data[i])-1):
            temp[i].append(sum(data[i][j+1])+1)
    return temp


# Draws a pHCurve given a 3D array across pHs and does a curve fit to that data
# Expects a histogram from each pH at a specific residue
def pHCurve(rawCountData, pHRange, outputDirName, saveType):
    residue = rawCountData[0][0][1]

    # Format Data
    rawCountData = sumAcrossTautomerStates(rawCountData)

    # Data Manipulation/Generation
    edgeCutoffData = []
    edgeCutoffFractions = []
    for i in range(len(rawCountData)):
        try:
            edgeCutoffData.append(rawCountData[i][0] / (rawCountData[i][0] + rawCountData[i][-1]))
            edgeCutoffFractions.append(rawCountData[i][0] / rawCountData[i][-1])
        except ZeroDivisionError:
            print("Could not form pH curve for " + residue + " due to a division by zero.")
            return

    edgeCutoffData = sorted(edgeCutoffData)
    pHRange = sorted(pHRange)
    print("Fractions (Dep / (Dep + Pro)): " + str(edgeCutoffData))
    print("pH Range: " + str(pHRange))
        
    finepHRange = np.linspace(min(pHRange), max(pHRange), 100)
    try:
        fitData, covariance = curve_fit(hendersonHasselBalch, pHRange, edgeCutoffData, bounds=([0.0,0.0],[5.0,14.0]))
    except RuntimeError:
        print("Fit Failed")
        return

    nValue = fitData[0]
    pKa = fitData[1]
    print("n = " + str(nValue))
    print("pKa = " + str(pKa))
    fit = generateData([nValue], finepHRange, pKa)

    # Plotting
    fig, ax = plt.subplots()

    for dataSet in fit:
        ax.plot(finepHRange, dataSet, linewidth=2.5)

    ax.plot(pHRange, edgeCutoffData, linewidth=3, label=residue + " data")
    ax.plot(finepHRange, fit[0], linewidth=3, label="Fit")
    ax.legend()

    ax.set(xlim=(8.4, 11.901), xticks=np.arange(8.4, 12, .5),
           ylim=(0, 1.001), yticks=np.arange(0, 1.1, .1))

    plt.xlabel("pH")
    plt.ylabel("Proportion of unprotonated states")

    plt.grid()
    plt.show()
    plt.savefig(outputDirName + residue + "Curve" + saveType)


# Draws a 3D histogram given a 2D array
# Expects a 2D hist with the header
def histogram(hist, outputDirName, saveType):
    residue = hist[0][0]
    pH = hist[0][-1]
    hist = np.array(hist[1:])
    
    fig = plt.figure(figsize=(16, 12), dpi=160)
    ax = plt.axes(projection='3d')
    ax.view_init(20, 230)
    x = np.array([[i] * 10 for i in range(10)]).ravel()  # x coordinates of each bar
    y = np.array([i for i in range(10)] * 10)  # y coordinates of each bar
    x = x * 0.1
    y = y * 0.1

    z = np.zeros(100)  # z coordinates of each bar
    dx = np.ones(100) * 0.1  # length along x-axis of each bar
    dy = np.ones(100) * 0.1  # length along y-axis of each bar
    dz = hist.ravel()  # length along z-axis of each bar (height)
    
    colors = plt.cm.viridis_r(hist.flatten() / float(hist.max()))
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='max')
    ax.set_title(residue + ' Populations at pH = ' + pH, y=1.0, pad=-14, fontsize=24)
    ax.set_ylabel("Tautomer Lambda", fontsize=20)
    ax.set_xlabel("Titration Lambda", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(20)
    plt.show()
    plt.savefig(outputDirName + residue + "AtPh" + pH + saveType)


# Returns a 3D array where each 2D array is a histogram at one residue at one pH (all pH's are the same)
def readESVs(filepath, pHs):
    file = open(filepath)
    temp = file.readline()

    histograms = []
    ESVNum = 0
    while temp is not "":
        if temp.__contains__('pH:'):
            data = temp.split()
            if ESVNum == 0:
                pHs.append(float(data[-1]))
            file.readline()
            file.readline()
            temp = file.readline()
            histograms.append([])
            histograms[ESVNum].append(data)
            for i in range(10):
                data = temp.split()[1:]
                for i in range(len(data)):
                    data[i] = int(data[i])
                if len(data) > 1:
                    histograms[ESVNum].append(data)
                temp = file.readline()
            ESVNum += 1
        temp = file.readline()

    return histograms


# Returns a boolean indicating whether the program has access to the correct directories with .esv files
def checkDirectories(currentDir, sep):
    # Find the number of windows
    maxWindow = False
    window = 0
    while not maxWindow:
        if os.path.isdir(currentDir + sep + str(window)):
            window += 1
        else:
            window -= 1
            maxWindow = True
    print('Number of widows to sort: ' + str(window + 1))

    if window <= 0:
        print('Nothing to sort')
        quit()

    return window


# Returns the base name of the first esv file it finds in the given search directory
def checkForESV(searchDirectory):
    for i in os.listdir(searchDirectory):
        if i.__contains__("penta.esv"):  # TODO: May have to change this depending on pdb file name (searching for just esv will cause an error because of the backup esvs)
            return i.replace(".esv", "")


if __name__ == "__main__":
    # Initial Checks
    separator = "/"
    cwd = os.getcwd()
    print("Current Working Directory: " + cwd)
    maxWindowIndex = checkDirectories(cwd, separator)

    names = []
    for i in range(maxWindowIndex + 1):
        names.append(checkForESV(cwd + separator + str(i)))

    if len(set(names)) != 1 or len(names) != (maxWindowIndex+1):
        print("Did not find the .esv file in all of the directories. Do some cleaning and run again. "
              "Program ending...")
        quit()

    baseName = names[0]
    filename = baseName + ".esv"
    print("ESV file being read from each directory: " + baseName + ".esv")
    # Data Collection
    windows = []
    pHs = []
    for i in range(maxWindowIndex + 1):
        windows.append(readESVs(cwd + separator + str(i) + separator + filename, pHs))

    # Make new directories
    histogramsDirName = "histograms"
    curvesDirName = "pHCurves"
    try:
        os.mkdir(cwd + separator + histogramsDirName)
    except FileExistsError:
        print("Histograms directory already exists.")
    try:
        os.mkdir(cwd + separator + curvesDirName)
    except FileExistsError:
        print("Curves directory already exists.")

    for i in range(len(windows[0])):
        try:
            os.mkdir(cwd + separator + histogramsDirName + separator + windows[0][i][0][1])
        except FileExistsError:
            print(histogramsDirName + separator + windows[0][i][0][1] + " directory already exists.")

    saveAsFileType = ".jpg"

    # Histograms
    # for i in windows:
        # for j in i:
            # histogram(j, histogramsDirName + separator + j[0][1] + separator, saveAsFileType)

    # pH curves
    for i in range(len(windows[0])):
        curveData = []
        for j in range(len(windows)):
            curveData.append(windows[j][i])

        print()
        print(str(curveData[0][0][1]) + " pH curve data and predictions: ")
        if curveData[0][0][1].__contains__("HIS"):
            curveHID = copy.deepcopy(curveData)
            curveHIE = copy.deepcopy(curveData)
            curveHIE[0][0][0] = "HIE"
            curveHID[0][0][0] = "HID"

            for i in range(len(curveHIE)): # window
                for j in range(len(curveHIE[i])-2): # histogram
                    for k in range(len(curveHIE[i][j+1])): # numbers
                        if k != len(curveHIE[i][j+1])-1:
                            curveHIE[i][j+1][k] = 0
                        if k != 0:
                            curveHID[i][j+1][k] = 0
            print("HIE predictions: ")
            pHCurve(curveHIE, pHs, curvesDirName + separator, saveAsFileType)
            print("\nHID predictions: ")
            pHCurve(curveHID, pHs, curvesDirName + separator, saveAsFileType)
        else:
            pHCurve(curveData, pHs, curvesDirName + separator, saveAsFileType)


