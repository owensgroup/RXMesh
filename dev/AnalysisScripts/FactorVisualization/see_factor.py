import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Define the paths to the MTX files
base_path = "/home/behrooz/Desktop/Last_Project/RXMesh-dev/output"
mtx_files = {
    "METIS": os.path.join(base_path, "factor_METIS.mtx"),
    # "NEUTRAL": os.path.join(base_path, "factor_NEUTRAL.mtx"),
    "RXMesh_ND": os.path.join(base_path, "factor_RXMesh_ND.mtx")
}

# Output directory for figures
output_dir = os.path.join(base_path, "figures")
os.makedirs(output_dir, exist_ok=True)

# Process each matrix
for name, filepath in mtx_files.items():
    print(f"Loading {name} matrix from {filepath}...")
    
    # Load the matrix
    matrix = sio.mmread(filepath)
    
    # Get matrix info
    nnz = matrix.nnz
    rows, cols = matrix.shape
    
    print(f"  Matrix size: {rows} x {cols}")
    print(f"  Non-zeros: {nnz}")
    print(f"  Sparsity: {100 * (1 - nnz / (rows * cols)):.2f}%")
    
    # Create spy plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spy(matrix, markersize=0.5, color='black')
    ax.set_title(f'Sparsity Pattern: {name}\n{rows}x{cols}, nnz={nnz}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"spy_{name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved figure to {output_file}")
    
    plt.close(fig)

# Create a comparison plot with all three matrices
print("\nCreating comparison plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (name, filepath) in enumerate(mtx_files.items()):
    matrix = sio.mmread(filepath)
    axes[idx].spy(matrix, markersize=0.3, color='black')
    axes[idx].set_title(f'{name}\nnnz={matrix.nnz}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Column', fontsize=10)
    axes[idx].set_ylabel('Row', fontsize=10)

plt.tight_layout()
comparison_file = os.path.join(output_dir, "spy_comparison.png")
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"Saved comparison figure to {comparison_file}")

plt.close(fig)

print("\nDone! All figures saved successfully.")

