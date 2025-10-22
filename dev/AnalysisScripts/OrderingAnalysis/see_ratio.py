import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_path = "/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/ordering_benchmark/sep_runtime_analysis.csv"
df = pd.read_csv(csv_path)

# Get unique meshes and sort by G_N
mesh_info = df[['mesh_name', 'G_N']].drop_duplicates().sort_values('G_N')
meshes_sorted = mesh_info['mesh_name'].tolist()

# Prepare data for plotting
metis_ratios = []
rxmesh_ratios = []
best_poc_ratios = []
best_max_degree_ratios = []
best_basic_ratios = []

for mesh in meshes_sorted:
    mesh_data = df[df['mesh_name'] == mesh]
    
    # Get METIS fill-ratio
    metis_row = mesh_data[mesh_data['ordering_type'] == 'METIS']
    metis_ratio = metis_row['fill-ratio'].values[0] if len(metis_row) > 0 else np.nan
    metis_ratios.append(metis_ratio)
    
    # Get RXMESH_ND fill-ratio
    rxmesh_row = mesh_data[mesh_data['ordering_type'] == 'RXMesh_ND']
    rxmesh_ratio = rxmesh_row['fill-ratio'].values[0] if len(rxmesh_row) > 0 else np.nan
    rxmesh_ratios.append(rxmesh_ratio)
    
    # Get best POC_ND fill-ratio (minimum across all configurations)
    poc_data = mesh_data[mesh_data['ordering_type'] == 'POC_ND']
    best_poc_ratio = poc_data['fill-ratio'].min() if len(poc_data) > 0 else np.nan
    best_poc_ratios.append(best_poc_ratio)
    
    # Get best POC_ND with max_degree
    poc_max_degree = poc_data[poc_data['separator_finding_method'] == 'max_degree']
    best_max_degree = poc_max_degree['fill-ratio'].min() if len(poc_max_degree) > 0 else np.nan
    best_max_degree_ratios.append(best_max_degree)
    
    # Get best POC_ND with basic
    poc_basic = poc_data[poc_data['separator_finding_method'] == 'basic']
    best_basic = poc_basic['fill-ratio'].min() if len(poc_basic) > 0 else np.nan
    best_basic_ratios.append(best_basic)

# Convert to numpy arrays
metis_ratios = np.array(metis_ratios)
rxmesh_ratios = np.array(rxmesh_ratios)
best_poc_ratios = np.array(best_poc_ratios)
best_max_degree_ratios = np.array(best_max_degree_ratios)
best_basic_ratios = np.array(best_basic_ratios)

# Normalize by METIS
metis_normalized = metis_ratios / metis_ratios
rxmesh_normalized = rxmesh_ratios / metis_ratios
best_poc_normalized = best_poc_ratios / metis_ratios
best_max_degree_normalized = best_max_degree_ratios / metis_ratios
best_basic_normalized = best_basic_ratios / metis_ratios

# Figure 1: Overall Comparison (Best POC_ND, RXMESH_ND, METIS)
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(meshes_sorted))
width = 0.25

bars1 = ax1.bar(x - width, best_poc_normalized, width, label='Best POC_ND', color='#2ecc71')
bars2 = ax1.bar(x, rxmesh_normalized, width, label='RXMESH_ND', color='#e74c3c')
bars3 = ax1.bar(x + width, metis_normalized, width, label='METIS', color='#3498db')

ax1.set_xlabel('Mesh Name', fontsize=12)
ax1.set_ylabel('Normalized Fill-Ratio', fontsize=12)
ax1.set_title('Fill-Ratio Comparison (Normalized by METIS)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(meshes_sorted, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig1.savefig('/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/ordering_benchmark/overall_comparison.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved: overall_comparison.png")

# Figure 2: POC_ND Method Comparison (max_degree vs basic)
fig2, ax2 = plt.subplots(figsize=(12, 6))

width2 = 0.35

bars1 = ax2.bar(x - width2/2, best_max_degree_normalized, width2, label='Best max_degree', color='#9b59b6')
bars2 = ax2.bar(x + width2/2, best_basic_normalized, width2, label='Best basic', color='#f39c12')

ax2.set_xlabel('Mesh Name', fontsize=12)
ax2.set_ylabel('Normalized Fill-Ratio', fontsize=12)
ax2.set_title('POC_ND: max_degree vs basic (Normalized by METIS)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(meshes_sorted, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig2.savefig('/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/ordering_benchmark/poc_method_comparison.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved: poc_method_comparison.png")

plt.show()

