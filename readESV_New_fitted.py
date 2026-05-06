import os
import copy
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

T = 300       # system temperature (K) - constant

# Returns an 2D where each array are the coordinates of a HH curve
def generateData(n, x, pKa):
    result = hendersonHasselBalch(x, n, pKa)
    return result

# Returns the result of putting x into the HH with hill=n and pKa=pKa
def hendersonHasselBalch(x, n, pKa):
    superScript = n * (pKa - x)
    y = 1 / (1 + 10.0 ** superScript)  # math.pow is not vectorized
    return y

def pHCurve(rawCountData, residue, pHRange, outputDirName, saveType, Tlambda):
    # Reformat Data
    countData = rawCountData
    summedCountData = np.sum(countData, axis=2)

    eps = 1e-12
    edgeCutoffData = []
    edgeCutoffFractions = []
    for i in summedCountData:
        ndeprot_phafed = float(i[0]) if len(i) > 0 else 0.0
        nprot_phafed = float(i[-1]) if len(i) > 0 else 0.0

        # λ=0 is PROTONATED, λ=1 is DEPROTONATED
        #nprot_phafed   = float(i[0]) + float(i[1]) if len(i) > 0 else 0.0  # bins 0,1 = λ∈[0,0.2] = protonated
        #ndeprot_phafed = float(i[-1]) + float(i[-2]) if len(i) > 0 else 0.0 # bins 8,9 = λ∈[0.8,1.0] = deprotonated


        # Guard against zeros
        if nprot_phafed <= 0.0:
            nprot_phafed = eps
        if ndeprot_phafed <= 0.0:
           ndeprot_phafed = eps

        ratio_phafed = ndeprot_phafed / nprot_phafed
        exponent = (Tlambda * np.log(ratio_phafed) / T)
        ndeprot_over_nprot = np.exp(exponent)

        # fraction deprotonated (Eq. 9)
        S_deprot = ndeprot_over_nprot / (ndeprot_over_nprot + 1)

        edgeCutoffData.append(S_deprot)
        edgeCutoffFractions.append(ratio_phafed)

    # Sort pH values
    pHRange = sorted(pHRange)
    print("Fractions (Eq. 8 & 9 rescaled): " + str(edgeCutoffData))
    print("Ratios (raw counts): " + str(edgeCutoffFractions))
    print("pH Range: " + str(pHRange))

    finepHRange = np.linspace(0, 14, 500)
    try:
        print(len(pHRange))
        print(len(edgeCutoffData))
        fitData, covariance = curve_fit(hendersonHasselBalch, pHRange, edgeCutoffData,
                                        bounds=([0, -1], [2, 20]))
    except ValueError as err:
        print("Fit Failed")
        print(err)
        return None, None

    n = fitData[0]
    pKa = fitData[1]
    print("n = " + str(n))
    print("pKa = " + str(pKa))
    fit = hendersonHasselBalch(finepHRange, n, pKa)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(finepHRange, fit, "--", linewidth=1, label="Fit", color="blue")
    ax.scatter(pHRange, edgeCutoffData, linewidth=.5, label=residue + " datapoints", color="black")
    ax.set(xlim=(min(pHRange) - .25, max(pHRange) + .25), xticks=np.arange(min(pHRange), max(pHRange) + .5, .5),
           ylim=(-.1, 1.1), yticks=np.arange(0, 1.1, .1))
    plt.title(residue + " pH Curve")
    plt.xlabel("pH", fontsize=12)
    plt.ylabel("Fraction Deprotonated", fontsize=12)
    xPosition = .15
    yPosition = .82
    perr = np.sqrt(np.diag(covariance))
    plt.figtext(xPosition, yPosition, "pKa = %4.2f ± %4.2f" % (pKa, perr[1]), fontsize=12)
    plt.figtext(xPosition, yPosition - .05, "    n = %4.2f ± %4.2f" % (n, perr[0]), fontsize=12)
    plt.show()
    os.chdir(outputDirName)
    plt.savefig(residue + "." + saveType, format=saveType)
    os.chdir(os.pardir)
    plt.close(fig)
    
    return n, pKa

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
    while temp != "":
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
    print('Number of windows in ' + currentDir + ': ' + str(window + 1))

    if window <= 0:
        print('Nothing to sort in ' + currentDir + '')
        quit()

    return window


# Returns the base name of the first esv file it finds in the given search directory
def checkForESV(searchDirectory):
    for i in os.listdir(searchDirectory):
        if i.endswith(".esv") and "backup" not in i:
            return i.replace(".esv", "")
    print("Did not find an .esv file in " + searchDirectory + ". Ignoring this directory. ")
    return ""


if __name__ == "__main__":
    # Init
    separator = os.sep
    
    # Check for Tlambda argument and directory argument
    if len(sys.argv) < 3:
        print("Usage: python readESV_New_fitted.py <Tlambda> <directory>")
        print("Example: python readESV_New_fitted.py 750 750K/")
        quit()
    
    # First argument is Tlambda
    try:
        Tlambda = float(sys.argv[1])
    except ValueError:
        print("Error: Tlambda must be a number")
        quit()
    
    # Second argument is the directory to process
    target_dir = sys.argv[2]
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist")
        quit()
    
    print(f"Using Tlambda = {Tlambda} K")
    print(f"Processing directory: {target_dir}")
    
    # Change to the target directory
    original_dir = os.getcwd()
    os.chdir(target_dir)
    cwd = os.getcwd()
    print(f"Changed to: {cwd}")
    
    inputs = [cwd]  # Process the current directory
    
    windows = 0
    matchingWindows = True
    insert=False
    pdbName = ""
    allPhs = []
    headerLines = []
    allData = []
    ignoreDir = []

    # Read in directories
    for i in inputs:
        if not os.path.isdir(i):
            print("Invalid directory: " + i)
            quit()
        temp = checkDirectories(i, separator)
        if windows == 0:
            windows = temp

        print("Current Working Directory: " + os.getcwd())
        maxWindowIndex = checkDirectories(cwd, separator)

        names = []
        for j in range(maxWindowIndex + 1):
            names.append(checkForESV(cwd + separator + str(j)))

        if names.__contains__(""):
            for j in range(len(names)):
                if names[j] == "":
                    ignoreDir.append(j)
        print("Ignoring directories: " + str(ignoreDir))
        if len(ignoreDir) >= maxWindowIndex // 2:
            print("Too many directories ignored. Program ending...")
            quit()

        for j in names:
            if j != "" and pdbName == "":
                pdbName = j
                break
            elif j != "" and j != pdbName:
                print("Multiple PDB names found. Currently in directory: " + cwd)
                print("PDB names found: " + pdbName + ", " + j)
                pdbName = j
                break

        filename = pdbName + ".esv"
        print("ESV file being read from each directory in " + i + ": " + pdbName + ".esv")

        # Data Collection
        windows = []
        pHs = []
        for j in range(maxWindowIndex + 1):
            if j not in ignoreDir:
                windows.append(readESVs(cwd + separator + str(j) + separator + filename, pHs))
            else:
                numberOfESVs = len(windows[0])
                filler = []
                for k in range(numberOfESVs):
                    filler.append([])
                    for v in range(11):
                        filler[k].append([])
                        for _ in range(10):
                            filler[k][v].append(0)
                windows.append(filler)
                pHs.append(-1)
                insert = True

        allPhs.append(pHs)
        for j in range(len(windows)):
            for k in range(len(windows[j])):
                headerLines.append(windows[j][k][0])
                windows[j][k] = windows[j][k][1:]

        ignoreDir = []
        allData.append(windows)

    consensusPhs = []
    allPhsIndex = 0
    length = 0
    for i in range(len(allPhs)):
        if len(allPhs[i]) > length:
            length = len(allPhs[i])
            allPhsIndex = i
        consensusPhs = list(set(consensusPhs) | set(allPhs[i]))
    if insert:
        consensusPhs = consensusPhs[:-1]
    print("Consensus pHs: " + str(consensusPhs))
    print("Number of pHs: " + str(len(consensusPhs)))

    # Replace all -1 pH values with the consensus pH
    for i in range(len(allPhs)):
        for j in range(len(allPhs[i])):
            if allPhs[i][j] == -1:
                # Find the set difference between the consensus pHs and the pHs in the current directory
                difference = list(set(consensusPhs) - set(allPhs[i]))
                # Replace the difference into the current directory's pH list where the -1 was
                allPhs[i][j] = difference[0]

    # Convert allData to a numpy array
    allData = np.array(allData)

    # Sort allData by pH
    for i in range(len(allData)):
        temp = allData[i][np.argsort(allPhs[i])]
        # Check if temp and allData[i] are the same
        if np.array_equal(temp, allData[i]):
            print("Not Sorting")
        allData[i] = temp

    # Combine all data from the various outer directories into one array
    windows = np.sum(allData, axis=0)
    pHs = consensusPhs

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

    for i in range(windows.shape[1]):
        try:
            os.mkdir(cwd + separator + histogramsDirName + separator + headerLines[i][1])
        except FileExistsError:
            print(histogramsDirName + separator + headerLines[i][1] + " directory already exists.")
    saveAsFileType = "pdf"

    # Histograms
    # for i in windows:
    # for j in i:
    # histogram(j, histogramsDirName + separator + j[0][1] + separator, saveAsFileType)

    # pH curves - now passing Tlambda
    all_pKa_values = []
    for i in range(windows.shape[1]):
        print("")
        curveData = windows[:, i, :, :]
        print(str(headerLines[i][1]) + " pH curve data and predictions: ")
        n, pKa = pHCurve(curveData, headerLines[i][1], pHs, curvesDirName, saveAsFileType, Tlambda)
        if pKa is not None:
            all_pKa_values.append((headerLines[i][1], n, pKa))
    
    # Print summary
    print("\n" + "="*50)
    print("Summary of pKa values:")
    for residue, n, pKa in all_pKa_values:
        print(f"Residue: {residue}, n = {n:.3f}, pKa = {pKa:.3f}")
    
    # Change back to original directory
    os.chdir(original_dir)
