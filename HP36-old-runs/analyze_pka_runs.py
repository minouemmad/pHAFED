#!/usr/bin/env python3
"""
Analysis script for pKa calculations from multiple simulation runs.
Extracts pKa values, simulation parameters, and compares with experimental data.
"""

import subprocess
import re
import os
import pandas as pd
from pathlib import Path
import numpy as np

# Define experimental pKa values
EXPERIMENTAL_PKA = {
    '44-ASD': 3.10,
    '45-GLD': 4.00,
    '46-ASD': 3.50,
    '72-GLD': 4.40
}

# Target residues to analyze
TARGET_RESIDUES = ['44-ASD', '45-GLD', '46-ASD', '72-GLD']

def extract_parameters_from_job(job_file_path):
    """
    Extract thetaMass, thetaTemp, and thetaFriction from dynamic.job file.
    
    Args:
        job_file_path: Path to the dynamic.job file
        
    Returns:
        dict: Dictionary containing mass, temp, and friction parameters
    """
    params = {
        'thetaMass': None,
        'thetaTemp': None,
        'thetaFriction': None
    }
    
    try:
        with open(job_file_path, 'r') as f:
            content = f.read()
            
        # Look for the PhDynamics command line
        # Pattern to match --thetaMass, --thetaTemp, --thetaFriction
        mass_match = re.search(r'--thetaMass\s+(\d+)', content)
        temp_match = re.search(r'--thetaTemp\s+(\d+)', content)
        friction_match = re.search(r'--thetaFriction\s+(\d+)', content)
        
        if mass_match:
            params['thetaMass'] = int(mass_match.group(1))
        if temp_match:
            params['thetaTemp'] = int(temp_match.group(1))
        if friction_match:
            params['thetaFriction'] = int(friction_match.group(1))
            
    except FileNotFoundError:
        print(f"Warning: Job file not found: {job_file_path}")
    except Exception as e:
        print(f"Error reading job file {job_file_path}: {e}")
    
    return params

def parse_pka_output(output_text):
    """
    Parse the output from readESV_New_fitted.py to extract pKa values.
    
    Args:
        output_text: String output from the readESV script
        
    Returns:
        dict: Dictionary mapping residue names to pKa values
    """
    pka_values = {}
    
    # Split output into sections for each residue
    lines = output_text.split('\n')
    
    current_residue = None
    for i, line in enumerate(lines):
        # Look for residue headers like "44-ASD pH curve data and predictions:"
        residue_match = re.match(r'^(\d+-[A-Z]{3})\s+pH curve data', line)
        if residue_match:
            current_residue = residue_match.group(1)
        
        # Look for pKa value lines
        if current_residue and line.strip().startswith('pKa = '):
            pka_match = re.search(r'pKa = ([\d.]+)', line)
            if pka_match:
                pka_values[current_residue] = float(pka_match.group(1))
                current_residue = None  # Reset for next residue
    
    return pka_values

def run_analysis_for_directory(run_dir):
    """
    Run the readESV_New_fitted.py script for a given directory.
    
    Args:
        run_dir: Path to the run directory (e.g., 'run1')
        
    Returns:
        dict: Extracted pKa values
    """
    try:
        # Run the Python script and capture output
        result = subprocess.run(
            ['python', 'readESV_New_fitted.py', run_dir],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        pka_values = parse_pka_output(result.stdout)
        return pka_values
        
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis for {run_dir}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return {}
    except Exception as e:
        print(f"Unexpected error for {run_dir}: {e}")
        return {}

def calculate_accuracy(calculated_pka, experimental_pka):
    """
    Calculate accuracy as the absolute error between calculated and experimental pKa.
    
    Args:
        calculated_pka: Calculated pKa value
        experimental_pka: Experimental pKa value
        
    Returns:
        float: Absolute error
    """
    return abs(calculated_pka - experimental_pka)

def main():
    """Main analysis function."""
    
    # Define run directories
    run_dirs = [f'run{i}' for i in range(1, 6)]
    
    # Storage for all results
    all_results = []
    
    print("Starting pKa analysis for runs 1-5...\n")
    
    for run_dir in run_dirs:
        print(f"Processing {run_dir}...")
        
        # Check if directory exists
        if not os.path.isdir(run_dir):
            print(f"Warning: Directory {run_dir} not found. Skipping...")
            continue
        
        # Extract parameters from dynamic.job
        job_file = os.path.join(run_dir, '0', 'dynamic.job')
        params = extract_parameters_from_job(job_file)
        
        print(f"  Parameters: Mass={params['thetaMass']}, Temp={params['thetaTemp']}, Friction={params['thetaFriction']}")
        
        # Run the analysis
        pka_values = run_analysis_for_directory(run_dir)
        
        if not pka_values:
            print(f"  Warning: No pKa values extracted for {run_dir}")
            continue
        
        # Build result row
        result_row = {
            'Run': run_dir,
            'thetaMass': params['thetaMass'],
            'thetaTemp': params['thetaTemp'],
            'thetaFriction': params['thetaFriction']
        }
        
        # Add calculated pKa values and errors for target residues
        errors = []
        for residue in TARGET_RESIDUES:
            calc_pka = pka_values.get(residue, None)
            exp_pka = EXPERIMENTAL_PKA[residue]
            
            result_row[f'{residue}_Calculated'] = calc_pka
            result_row[f'{residue}_Experimental'] = exp_pka
            
            if calc_pka is not None:
                error = calculate_accuracy(calc_pka, exp_pka)
                result_row[f'{residue}_Error'] = error
                errors.append(error)
                print(f"  {residue}: Calculated={calc_pka:.2f}, Experimental={exp_pka:.2f}, Error={error:.2f}")
            else:
                result_row[f'{residue}_Error'] = None
                print(f"  {residue}: No calculated value found")
        
        # Calculate global accuracy (mean absolute error)
        if errors:
            global_error = np.mean(errors)
            result_row['Global_MAE'] = global_error
            print(f"  Global Mean Absolute Error: {global_error:.2f}")
        else:
            result_row['Global_MAE'] = None
        
        all_results.append(result_row)
        print()
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better presentation
    column_order = ['Run', 'thetaMass', 'thetaTemp', 'thetaFriction']
    
    for residue in TARGET_RESIDUES:
        column_order.extend([
            f'{residue}_Calculated',
            f'{residue}_Experimental',
            f'{residue}_Error'
        ])
    
    column_order.append('Global_MAE')
    
    # Reorder DataFrame columns
    df = df[column_order]
    
    # Save to Excel
    output_file = '/Dedicated/schnieders/maemmad/pHAFED/HP36/pKa_analysis_results.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write main results
        df.to_excel(writer, sheet_name='pKa Analysis', index=False)
        
        # Create a summary sheet
        summary_data = {
            'Residue': TARGET_RESIDUES,
            'Experimental pKa': [EXPERIMENTAL_PKA[res] for res in TARGET_RESIDUES]
        }
        
        # Add calculated pKa for each run
        for run_dir in run_dirs:
            run_data = df[df['Run'] == run_dir]
            if not run_data.empty:
                summary_data[f'{run_dir}_Calc'] = [
                    run_data[f'{res}_Calculated'].values[0] 
                    for res in TARGET_RESIDUES
                ]
                summary_data[f'{run_dir}_Error'] = [
                    run_data[f'{res}_Error'].values[0] 
                    for res in TARGET_RESIDUES
                ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add a parameters sheet
        params_df = df[['Run', 'thetaMass', 'thetaTemp', 'thetaFriction', 'Global_MAE']]
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"\nSummary:")
    print(df[['Run', 'Global_MAE']].to_string(index=False))
    
    return df

if __name__ == '__main__':
    main()
