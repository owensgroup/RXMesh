#!/bin/bash
# Script to run ordering analysis
cd /home/behrooz/Desktop/Last_Project/RXMesh-dev

# Mesh base directory
MESH_DIR="/media/behrooz/FarazHard/Last_Project/MIT_meshes"

# List of meshes to test (add/remove mesh names as needed)
MESHES=(
    "fish.obj"
    "violin.obj"
    "cloth10x10.obj"
    "nefertiti.obj"
    "488FancyRoller.obj"
    "Chevy_Corvette_Carbon65.obj"
    "SkysSkull_2_bone-smoothed-orientated.obj"
    "wingedvictory.obj"
)

BINARY="./cmake-build-release/bin/RXMesh_detailed_ordering_benchmark"

# Test each mesh with METIS, then POC_ND, then RXMESH_ND
for mesh in "${MESHES[@]}"; do
    # echo "Testing $mesh with METIS..."
    # $BINARY -i "$MESH_DIR/$mesh" -a METIS
    
    # Run POC_ND with different parameter combinations
    for l in metis amd; do
        #for t in max_degree basic; do
        for t in basic; do  
            #for r in patch_refinement redundancy_removal patch_redundancy_refinement nothing; do
            for r in bipartite_graph_refinement; do
                echo "Testing $mesh with POC_ND (l=$l, t=$t, r=$r)..."
                $BINARY -i "$MESH_DIR/$mesh" -a POC_ND -t $t -r $r -l $l
            done
        done
    done
    
    echo "Testing $mesh with RXMESH_ND..."
    $BINARY -i "$MESH_DIR/$mesh" -a RXMESH_ND
done

echo "All analysis completed!"
