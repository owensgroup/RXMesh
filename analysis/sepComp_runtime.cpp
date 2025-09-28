#include <iostream>
#include <parth/parth.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>


struct CLIArgs {
    std::string input_mesh;
    std::string output_address;

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Separator analysis"};

        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};



int main(int argc, char *argv[]) {
    // Load the mesh
    CLIArgs args(argc, argv);
    
    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input to specify the mesh file." << std::endl;
        return 1;
    }
    
    if (args.output_address.empty()) {
        std::cerr << "Error: Output folder not specified. Use -o or --output to specify the output folder." << std::endl;
        return 1;
    }
    
    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_address << std::endl;
    
    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh << std::endl;
        return 1;
    }

    //Create laplacian matrix
    Eigen::SparseMatrix<double> OL;
    igl::cotmatrix(OV, OF, OL);
  
    //init Parth
    std::vector<int> perm;
    PARTH::ParthAPI parth;
    parth.setMatrix(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), 1);
    parth.computePermutation(perm);

    for(int i = 0; i < parth.hmd.HMD_tree.size(); i++) {
      std::cout << "Separator time: " << parth.hmd.HMD_tree[i].separator_comp_time << std::endl;
      std::cout << "Permutation time: " << parth.hmd.HMD_tree[i].permute_time << std::endl;
    }
     
    
    return 0;
}