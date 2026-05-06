import subprocess
import re
import matplotlib.pyplot as plt
import os

# -----------------------------
# USER INPUTS
# -----------------------------
temps = [500, 750, 1000, 1250, 1500]
script = "readESV_New_fitted.py"

# Reference pKa (repex)
pKa_ref = 3.9

# Base directory where the temperature subdirectories are located
base_dir = "."  # Change this if your directories are elsewhere

# -----------------------------
# FUNCTIONS
# -----------------------------
def run_and_extract_pka(temp):
    # Construct the directory path
    dir_path = f"{temp}K/"
    
    # Check if directory exists
    if not os.path.isdir(dir_path):
        print(f"⚠️ Directory {dir_path} does not exist!")
        return None
    
    # Run the script with Tlambda as first argument and the directory as second argument
    # We're running from the parent directory, so we pass the directory path as an argument
    cmd = ["python", script, str(temp), dir_path]
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Print output for debugging (optional)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-2000:])  # Print last 2000 characters to avoid too much output
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Extract pKa using regex - looking for the pKa value from the fit
        output = result.stdout
        
        # Try multiple patterns to find pKa
        patterns = [
            r"pKa\s*=\s*([0-9.+-eE]+)",  # Original pattern
            r"pKa\s*=\s*([0-9.+-eE]+)\s*±",  # With error
            r"pKa\s*:\s*([0-9.+-eE]+)",  # Alternative format
        ]
        
        pka_value = None
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                pka_value = float(match.group(1))
                break
        
        # Also check the summary at the end
        if pka_value is None:
            # Look for summary line
            summary_match = re.search(r"Residue: \w+, n = [0-9.]+e?[+-]?[0-9]*, pKa = ([0-9.]+e?[+-]?[0-9]*)", output)
            if summary_match:
                pka_value = float(summary_match.group(1))
        
        if pka_value is not None:
            print(f"✓ Found pKa = {pka_value:.6f} for {temp}K")
        else:
            print(f"⚠️ Could not find pKa for {temp}K in output")
            
        return pka_value
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Process timed out for {temp}K")
        return None
    except Exception as e:
        print(f"⚠️ Error running script for {temp}K: {e}")
        return None


# -----------------------------
# MAIN
# -----------------------------
Tlambda_vals = []
delta_vals = []
pka_vals = []

for T in temps:
    pka = run_and_extract_pka(T)

    if pka is not None:
        delta = pKa_ref - pka

        Tlambda_vals.append(T)
        delta_vals.append(delta)
        pka_vals.append(pka)

        print(f"T = {T} K | pKa = {pka:.6f} | Δ = {delta:.6f}")

# -----------------------------
# PLOTTING
# -----------------------------
if Tlambda_vals:
    plt.figure(figsize=(10, 6))
    plt.scatter(Tlambda_vals, delta_vals, s=100, color='red', zorder=5)
    plt.plot(Tlambda_vals, delta_vals, 'b-', linewidth=2, alpha=0.7, zorder=3)
    
    # Add labels and title
    plt.xlabel(r"$T_\lambda$ (K)", fontsize=14)
    plt.ylabel(r"$\Delta(T_\lambda)$ = pKa$_{\text{ref}}$ - pKa$_{\text{AFED}}$", fontsize=14)
    plt.title("Aspartate (Penta) Δ vs Tλ", fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (T, delta) in enumerate(zip(Tlambda_vals, delta_vals)):
        plt.annotate(f"{delta:.3f}", 
                    (T, delta),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig("delta_vs_Tlambda.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # -----------------------------
    # PRINT TABLE
    # -----------------------------
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"{'Tλ (K)':<10} {'pKa':<15} {'Δ':<15}")
    print("-"*40)
    for T, pka, d in zip(Tlambda_vals, pka_vals, delta_vals):
        print(f"{T:<10} {pka:<15.6f} {d:<15.6f}")
    
    # Optionally save results to file
    with open("pka_results.txt", "w") as f:
        f.write("Tlambda_K\tpKa\tDelta\n")
        for T, pka, d in zip(Tlambda_vals, pka_vals, delta_vals):
            f.write(f"{T}\t{pka:.6f}\t{d:.6f}\n")
    print(f"\nResults also saved to pka_results.txt")
    
else:
    print("\nNo valid pKa values were extracted!")
