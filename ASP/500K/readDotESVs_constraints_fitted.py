import os
import copy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Configuration (NEW)
# =========================
T = 300.0        # >>> CHANGED: System temperature (K)
Tlambda = 500.0  # >>> CHANGED: Lambda temperature (K)
EPS = 1e-12      # >>> NEW: Small epsilon to avoid log/div-zero


# =========================
# HH Model (unchanged)
# =========================
def generateData(listOfN, listOfpH, pKa):
    result = []
    for n in range(len(listOfN)):
        result.append([])
        for pH in listOfpH:
            result[n].append(hendersonHasselBalch(pH, listOfN[n], pKa))
    return result


def hendersonHasselBalch(x, n, pKa):
    # y = 1 / (1 + 10^(n * (pKa - x)))
    superScript = n * (pKa - x)
    y = 1.0 / (1.0 + 10.0 ** superScript)
    return y


# =========================
# Helpers
# =========================
def sumAcrossTautomerStates(data):
    """
    Returns a 2D array of counts per pH, summed across tautomer lambda.
    Expects a 3D array of histograms (including header on each histogram).
    Shape in: [num_pH][11 rows (1 header + 10 grid rows)][10 columns]
    Shape out: [num_pH][10] sums across each of the 10 titration-λ bins
    """
    temp = []
    for i in range(len(data)):
        rows = data[i]
        values = []
        # rows[0] is header; rows[1:] are 10 lines of grid
        for j in range(len(rows) - 1):
            values.append(sum(rows[j + 1]))
        temp.append(values)
    return temp


def _deprot_fraction_from_edges(n_deprot_edge, n_prot_edge, T, Tlambda, eps=EPS):
    """
    Implements Eq. 8 & 9 rescaling from your updated logic:
    - raw ratio = ndeprot / nprot
    - rescaled ratio at system temp = exp( (Tlambda/T) * ln(raw_ratio) )
    - S_deprot = rescaled_ratio / (1 + rescaled_ratio)
    """
    nde = max(float(n_deprot_edge), eps)
    npr = max(float(n_prot_edge), eps)
    raw_ratio = nde / npr
    nde_over_npr_rescaled = np.exp((Tlambda / T) * np.log(raw_ratio))
    S_deprot = nde_over_npr_rescaled / (1.0 + nde_over_npr_rescaled)
    return S_deprot, raw_ratio


# =========================
# Plot / Fit pH curve (UPDATED)
# =========================
def pHCurve(rawCountData, pHRange, outputDirName, saveType):
    """
    rawCountData: list over pH windows; each entry is a histogram (with header)
    pHRange: list of pH values aligned with rawCountData
    """
    residue = rawCountData[0][0][1]  # from header [0] of first histogram

    # 1) Sum counts across tautomer lambda states
    counts_per_pH = sumAcrossTautomerStates(rawCountData)   # shape [num_pH][10]

    # 2) Use only the edge titration-λ bins: index 0 (deprot edge) and -1 (prot edge)
    S_deprot_list = []
    ratio_list = []
    for i in range(len(counts_per_pH)):
        ndeprot_edge = counts_per_pH[i][0]
        nprot_edge   = counts_per_pH[i][-1]
        S_deprot, raw_ratio = _deprot_fraction_from_edges(ndeprot_edge, nprot_edge, T, Tlambda, EPS)
        S_deprot_list.append(S_deprot)
        ratio_list.append(raw_ratio)

    # 3) Pair-sort by pH to keep alignment (avoid separate sorts)
    pairs = sorted(zip(pHRange, S_deprot_list), key=lambda t: t[0])
    pH_sorted = np.array([p for p, _ in pairs], dtype=float)
    S_sorted  = np.array([s for _, s in pairs], dtype=float)

    print(f"Residue: {residue}")
    print("Ratios (raw counts ndeprot/nprot): " + np.array2string(np.array(ratio_list), precision=5))
    print("Fractions (Eq. 8 & 9 rescaled): " + np.array2string(S_sorted, precision=5))
    print("pH Range (sorted): " + np.array2string(pH_sorted, precision=3))

    # 4) Fit HH model (n, pKa) to the rescaled fractions
    finepHRange = np.linspace(min(pH_sorted), max(pH_sorted), 400)
    try:
        popt, pcov = curve_fit(
            hendersonHasselBalch,
            pH_sorted, S_sorted,
            p0=(1.0, 7.0),
            bounds=([0.0, 0.0], [5.0, 14.0])
        )
        nValue, pKa = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        print(f"n = {nValue:.4f} ± {perr[0]:.4f}")
        print(f"pKa = {pKa:.4f} ± {perr[1]:.4f}")
    except Exception as e:
        print("Fit Failed:", e)
        return

    fit = generateData([nValue], finepHRange, pKa)

    # 5) Plot
    fig, ax = plt.subplots()
    ax.scatter(pH_sorted, S_sorted, linewidth=1.5, color="black", label=f"{residue} data")
    ax.plot(finepHRange, fit[0], "--", linewidth=2, color="blue", label="Fit (HH)")

    ax.set_xlim(min(pH_sorted) - 0.25, max(pH_sorted) + 0.25)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(np.floor(min(pH_sorted)), np.ceil(max(pH_sorted)) + 0.5, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Fraction Deprotonated (rescaled)", fontsize=12)
    ax.set_title(f"{residue} pH Curve")
    ax.grid(alpha=0.25)
    ax.legend()

    # Add fit text incl. temps
    xPosition = 0.15
    yPosition = 0.82
    plt.figtext(xPosition, yPosition,   f"pKa = {pKa:4.2f} ± {perr[1]:4.2f}", fontsize=12)
    plt.figtext(xPosition, yPosition-.05, f"n   = {nValue:4.2f} ± {perr[0]:4.2f}", fontsize=12)
    plt.figtext(xPosition, yPosition-.10, f"T = {T:.0f} K, Tλ = {Tlambda:.0f} K", fontsize=11)

    plt.show()
    # Save
    os.makedirs(outputDirName, exist_ok=True)
    out_path = os.path.join(outputDirName, f"{residue}Curve{saveType}")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 3D Histogram (unchanged)
# =========================
def histogram(hist, outputDirName, saveType):
    residue = hist[0][0]
    pH = hist[0][-1]
    hist = np.array(hist[1:])

    fig = plt.figure(figsize=(16, 12), dpi=160)
    ax = plt.axes(projection='3d')
    ax.view_init(20, 230)
    x = np.array([[i] * 10 for i in range(10)]).ravel()  # x coordinates of each bar
    y = np.array([i for i in range(10)] * 10)            # y coordinates of each bar
    x = x * 0.1
    y = y * 0.1

    z = np.zeros(100)
    dx = np.ones(100) * 0.1
    dy = np.ones(100) * 0.1
    dz = hist.ravel()

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
    os.makedirs(outputDirName, exist_ok=True)
    plt.savefig(os.path.join(outputDirName, residue + "AtPh" + pH + saveType))


# =========================
# ESV Reader (minor fix)
# =========================
def readESVs(filepath, pHs):
    file = open(filepath)
    temp = file.readline()

    histograms = []
    ESVNum = 0
    while temp != "":  # >>> CHANGED: avoid `is not` for strings
        if 'pH:' in temp:
            data = temp.split()
            if ESVNum == 0:
                pHs.append(float(data[-1]))
            file.readline()
            file.readline()
            temp = file.readline()
            histograms.append([])
            histograms[ESVNum].append(data)
            for _ in range(10):
                data = temp.split()[1:]
                for i in range(len(data)):
                    data[i] = int(data[i])
                if len(data) > 1:  # >>> CHANGED: replaced &gt; with >
                    histograms[ESVNum].append(data)
                temp = file.readline()
            ESVNum += 1
        temp = file.readline()

    return histograms


# =========================
# Directory Utilities
# =========================
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
    print('Number of windows to sort: ' + str(window + 1))

    if window <= 0:
        print('Nothing to sort')
        quit()

    return window


def checkForESV(searchDirectory):
    for i in os.listdir(searchDirectory):
        if i.__contains__("penta.esv"):  # NOTE: may need to adjust for your PDB base
            return i.replace(".esv", "")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Initial Checks
    separator = "/"
    cwd = os.getcwd()
    print("Current Working Directory: " + cwd)
    maxWindowIndex = checkDirectories(cwd, separator)

    names = []
    for i in range(maxWindowIndex + 1):
        names.append(checkForESV(cwd + separator + str(i)))

    if len(set(names)) != 1 or len(names) != (maxWindowIndex + 1):
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
    os.makedirs(cwd + separator + histogramsDirName, exist_ok=True)
    os.makedirs(cwd + separator + curvesDirName, exist_ok=True)

    for i in range(len(windows[0])):
        os.makedirs(cwd + separator + histogramsDirName + separator + windows[0][i][0][1], exist_ok=True)

    saveAsFileType = ".jpg"

    # Histograms (optional)
    # for i in windows:
    #     for j in i:
    #         histogram(j, histogramsDirName + separator + j[0][1] + separator, saveAsFileType)

    # pH curves
    for i in range(len(windows[0])):  # iterate residues
        curveData = []
        for j in range(len(windows)):  # iterate windows/pH
            curveData.append(windows[j][i])

        print()
        print(str(curveData[0][0][1]) + " pH curve data and predictions: ")

        if "HIS" in curveData[0][0][1]:
            # Split HID/HIE by zeroing the opposite endpoint across grids, as before
            curveHID = copy.deepcopy(curveData)
            curveHIE = copy.deepcopy(curveData)
            curveHIE[0][0][0] = "HIE"
            curveHID[0][0][0] = "HID"

            for wi in range(len(curveHIE)):             # window
                for hj in range(len(curveHIE[wi]) - 2): # histogram rows (excluding header + last?)
                    for k in range(len(curveHIE[wi][hj + 1])):  # counts
                        if k != len(curveHIE[wi][hj + 1]) - 1:
                            curveHIE[wi][hj + 1][k] = 0
                        if k != 0:
                            curveHID[wi][hj + 1][k] = 0

            print("HIE predictions: ")
            pHCurve(curveHIE, pHs, cwd + separator + curvesDirName + separator, saveAsFileType)
            print("\nHID predictions: ")
            pHCurve(curveHID, pHs, cwd + separator + curvesDirName + separator, saveAsFileType)
        else:
            pHCurve(curveData, pHs, cwd + separator + curvesDirName + separator, saveAsFileType)
