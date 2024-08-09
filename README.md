# **RXMesh** [![Ubuntu](https://github.com/owensgroup/RXMesh/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/owensgroup/RXMesh/actions/workflows/Ubuntu.yml) [![Windows](https://github.com/owensgroup/RXMesh/actions/workflows/Windows.yml/badge.svg)](https://github.com/owensgroup/RXMesh/actions/workflows/Windows.yml)

<p align="center">
    <img src="./assets/david_pacthes.png" width="80%"><br>
</p>

## **Contents**
- [**About**](#about)
- [**Compilation**](#compilation)
  * [**Dependencies**](#dependencies)
- [**Organization**](#organization)
- [**Programming Model**](#programming-model)  
  * [**Structures**](#structures)
  * [**Computation**](#computation)
  * [**Viewer**](#viewer)
  * [**Matrices and Vectors**](#matrices-and-vectors)
- [**Replicability**](#replicability)
- [**Bibtex**](#bibtex)

## **About**
RXMesh is a surface triangle mesh data structure and programming model for processing static meshes on the GPU. RXMesh aims at provides a high-performance, generic, and compact data structure that can handle meshes regardless of their quality (e.g., non-manifold). The programming model helps to hide the complexity of the data structure and provides an intuitive access model for different use cases. For more details, please check out our paper and GTC talk:

- *[RXMesh: A GPU Mesh Data Structure](https://escholarship.org/uc/item/8r5848vp)*<br>
*[Ahmed H. Mahmoud](https://www.ece.ucdavis.edu/~ahdhn/), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)*<br>
*[ACM Transaction on Graphics](https://dl.acm.org/doi/abs/10.1145/3450626.3459748) (Proceedings of SIGGRAPH 2021)*

- *[RXMesh: A High-performance Mesh Data Structure and Programming Model on the GPU  [S41051]](https://www.nvidia.com/gtc/session-catalog/?tab.scheduledorondemand=1583520458947001NJiE&search=rxmesh#/session/1633891051385001Q9SE)—NVIDIA GTC 2022*

The library also features a sparse and dense matrix infrastructure that is tightly coupled with the mesh data structure. We expose various [cuSolver](https://docs.nvidia.com/cuda/cusolver/index.html), [cuSparse](https://docs.nvidia.com/cuda/cusparse/), and [cuBlas](https://docs.nvidia.com/cuda/cublas/) operations through the sparse and dense matrices, tailored for geometry processing applications.

This repository provides 1) source code to reproduce the results presented in the paper (git tag [`v0.1.0`](https://github.com/owensgroup/RXMesh/tree/v0.1.0)) and 2) ongoing development of RXMesh.

## **Compilation**
The code can be compiled on Ubuntu, Windows, and WSL providing that CUDA (>=11.1.0) is installed. To run the executable(s), an NVIDIA GPU should be installed on the machine.

### **Dependencies**
- [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh) to verify the applications against reference CPU implementation
- [RapidJson](https://github.com/Tencent/rapidjson) to report the results in JSON file(s)
- [GoogleTest](https://github.com/google/googletest) for unit tests
- [spdlog](https://github.com/gabime/spdlog) for logging
- [glm](https://github.com/g-truc/glm.git) for small vectors and matrices operations 
- [Eigen](https://gitlab.com/libeigen/eigen) for small vectors and matrices operations 
- [Polyscope ](https://github.com/nmwsharp/polyscope) for visualization  
- [cereal](https://github.com/USCiLab/cereal.git) for serialization 


All the dependencies are installed automatically! To compile the code:

```
> git clone https://github.com/owensgroup/RXMesh.git
> cd RXMesh
> mkdir build 
> cd build 
> cmake ../
```
Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system.

## **Organization**
RXMesh is a CUDA/C++ header-only library. All unit tests are under the `tests/` folder. This includes the unit test for some basic functionalities along with the unit test for the query operations. All applications are under the `apps/` folder.

## **Programming Model**
The goal of defining a programming  model is to make it easy to write applications using RXMesh without getting into the nuances of the data structure. Applications written using RXMesh are composed of one or more of the high-level building blocks defined under [**Computation**](#computation). To use these building blocks, the user would have to interact with data structures specific to RXMesh discussed under [**Structures**](#structures). Finally, RXMesh integrates [Polyscope](https://polyscope.run) as a mesh [**Viewer**](#viewer) which the user can use to render their final results or for debugging purposes. 

### **Structures**
- **Attributes** are the metadata (geometry information) attached to vertices, edges, or faces. Allocation of the attributes is per-patch basis and managed internally by RXMesh. The allocation could be done on the host, device, or both. Allocating attributes on the host is only beneficial for I/O operations or initializing attributes and then eventually moving them to the device. 
  - Example: allocation
    ```c++
    RXMeshStatic rx("input.obj");
    auto vertex_color = 
      rx.add_vertex_attribute<float>("vColor", //Unique name 
                                     3,        //Number of attribute per vertex 
                                     DEVICE,   //Allocation place 
                                     SoA);     //Memory layout (SoA vs. AoS)                                 

    ```
  - Example: reading from `std::vector`
    ```c++
    RXMeshStatic rx("input.obj");
    std::vector<std::vector<float>> face_color_vector;
    //....

    auto face_color = 
      rx.add_face_attribute<int>(face_color_vector,//Input attribute where number of attributes per face is inferred 
                                 "fColor",         //Unique name                                
                                 SoA);             //Memory layout (SoA vs. AoS)                                  
    ```
  - Example: move, reset, and copy
    ```c++    
    //By default, attributes are allocated on both host and device     
    auto edge_attr = rx.add_edge_attribute<float>("eAttr", 1);  
    //Initialize edge_attr on the host 
    // ..... 

    //Move attributes from host to device 
    edge_attr.move(HOST, DEVICE);

    //Reset all entries to zero
    edge_attr.reset(0, DEVICE);

    auto edge_attr_1 = rx.add_edge_attribute<float>("eAttr1", 1);  

    //Copy from another attribute. 
    //Here, what is on the host sde of edge_attr will be copied into the device side of edge_attr_1
    edge_attr_1.copy_from(edge_attr, HOST, DEVICE);
    ```

- **Handles** are the unique identifiers for vertices, edges, and faces. They are usually internally populated by RXMesh (by concatenating the patch ID and mesh element index within the patch). Handles can be used to access attributes, `for_each` operations, and query operations. 

  - Example: Setting vertex attribute using vertex handle 
    ```c++  
    auto vertex_color = ...    
    VertexHandle vh; 
    //...
    
    vertex_color(vh, 0) = 0.9;
    vertex_color(vh, 1) = 0.5;
    vertex_color(vh, 2) = 0.6;
    ```

- **Iterators** are used during query operations to iterate over the output of the query operation. The type of iterator defines the type of mesh element iterated on e.g., `VertexIterator` iterates over vertices which is the output of `VV`, `EV`, or `FV` query operations. Since query operations are only supported on the device, iterators can be only used inside the GPU kernel. Iterators are usually populated internally. 

  - Example: Iterating over faces 
      ```c++  
      FaceIterator f_iter; 
      //...

      for (uint32_t f = 0; f < f_iter.size(); ++f) {	
        FaceHandle fh = f_iter[f];
        //do something with fh ....
      }
      ```


### **Computation**
- **`for_each`** runs a computation over all vertices, edges, or faces _without_ requiring information from neighbor mesh elements. The computation that runs on each mesh element is defined as a lambda function that takes a handle as an input. The lambda function could run either on the host, device, or both. On the host, we parallelize the computation using OpenMP. Care must be taken for lambda function on the device since it needs to be annotated using `__device__` and it can only capture by value. More about lambda function in CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda)
  - Example: using `for_each` to initialize attributes 
    ```cpp
    RXMeshStatic rx("input.obj");
    auto vertex_pos   = rx.get_input_vertex_coordinates();                   //vertex position 
    auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3, DEVICE); //vertex color 

    //This function will be executed on the device 
    rx.for_each_vertex(
        DEVICE,
        [vertex_color, vertex_pos] __device__(const VertexHandle vh) {
            vertex_color(vh, 0) = 0.9;
            vertex_color(vh, 1) = vertex_pos(vh, 1);
            vertex_color(vh, 2) = 0.9;
        });
    ```
  Alternatively, `for_each` operations could be written the same way as Queries operations (see below). This might be useful if the user would like to combine a `for_each` with queries operations in the same kernel. For more examples, checkout [`ForEach`](/tests/RXMesh_test/test_for_each.cuh) unit test. 

- **Queries** operations supported by RXMesh with description are listed below 

  | Query |  Description                               |
  |-------|:-------------------------------------------|
  | `VV`  | For vertex V, return its adjacent vertices |
  | `VE`  | For vertex V, return its incident edges    |
  | `VF`  | For vertex V, return its incident faces    |
  | `EV`  | For edge E, return its incident vertices   |
  | `EF`  | For edge E, return its incident faces      |
  | `FV`  | For face F, return its incident vertices   |
  | `FE`  | For face F, return its incident edges      |
  | `FF`  | For face F, return its adjacent faces      |

  Queries are only supported on the device. RXMesh API for queries takes a lambda function along with the type of query. The lambda function defines the computation that will be run on the query output. 

  - Example: [vertex normal computation](./apps/VertexNormal/vertex_normal_kernel.cuh)
    ```cpp
    template<uint32_t blockSize>
    __global__ void vertex_normal (Context context){      
	    auto compute_vn = [&](const FaceHandle face_id, const VertexIterator& fv) {
        	//This thread is assigned to face_id

        	// get the face's three vertices coordinates
        	vec3<T> c0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        	vec3<T> c1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
	        vec3<T> c2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

          //compute face normal
          vec3<T> n = cross(c1 - c0, c2 - c0);

	        // add the face's normal to its vertices
        		for (uint32_t v = 0; v < 3; ++v)     // for every vertex in this face
	            for (uint32_t i = 0; i < 3; ++i)   // for the vertex 3 coordinates
        		        atomicAdd(&normals(fv[v], i), n[i]);          
	    };

      //Query must be called by all threads in the block. Thus, we create this cooperative_group
      //that uses all threads in the block and pass to the Query 
      auto block = cooperative_groups::this_thread_block();
      
      Query<blockThreads> query(context);

      //Qeury will first perform the query, store the results in shared memory. ShmemAllocator is 
      //passed to the function to make sure we don't over-allocate or overwrite user-allocated shared
      //memory 
      ShmemAllocator shrd_alloc;

      //Finally, we run the user-defined computation i.e., compute_vn
      query.dispatch<Op::FV>(block, shrd_alloc, compute_vn);
    } 
    ```
  To save computation, `query.dispatch` could be run on a subset of the input mesh element i.e., _active set_. The user can define the active set using a lambda function that returns true if the input mesh element is in the active set. 

  - Example: defining active set
    ```cpp
    template<uint32_t blockSize>
    __global__ void active_set_query (Context context){
      auto active_set = [&](FaceHandle face_id) -> bool{ 
        // ....         
	    };

	    auto computation = [&](const FaceHandle face_id, const VertexIterator& fv) {          
        // ....         
	    };

	    query.dispatch<Op::FV, blockSize>(context, computation, active_set);
    } 
    ```

- **Reduction** operations apply a binary associative operation on the input attributes. RXMesh provides dot products between two attributes (of the same type), L2 norm of an input attribute, and user-defined reduction operation on an input attribute. For user-defined reduction operation, the user needs to pass a binary reduction functor with member `__device__ T operator()(const T &a, const T &b)` or use on of [CUB's thread operators](https://github.com/NVIDIA/cub/blob/main/cub/thread/thread_operators.cuh) e.g., `cub::Max()`. Reduction operations require allocation of temporary buffers which we abstract away using `ReduceHandle`. 

  - Example: dot product, L2 norm, user-defined reduction 
    ```cpp 
    RXMeshStatic rx("input.obj");
    auto vertex_attr1 = rx.add_vertex_attribute<float>("v_attr1", 3, DEVICE);
    auto vertex_attr2 = rx.add_vertex_attribute<float>("v_attr2", 3, DEVICE);

    // Populate vertex_attr1 and vertex_attr2 
    //....

    //Reduction handle 
    ReduceHandle reduce(v1_attr);

    //Dot product between two attributes. Results are returned on the host 
    float dot_product = reduce.dot(v1_attr, v2_attr);

    cudaStream_t stream; 
    //init stream 
    //...

    //Reduction operation could be performed on specific attribute and using specific stream 
    float l2_norm = reduce.norm2(v1_attr, //input attribute 
                                 1,       //attribute ID. If not specified, reduction is run on all attributes 
                                 stream); //stream used for reduction. 
    

    //User-defined reduction operation 
    float l2_norm = reduce.reduce(v1_attr,                               //input attribute 
                                  cub::Max(),                            //binary reduction functor 
                                  std::numeric_limits<float>::lowest()); //initial value 
    ```

### **Viewer**
Starting v0.2.1, RXMesh integrates [Polyscope](https://polyscope.run) as a mesh viewer. To use it, make sure to turn on the CMake parameter `USE_POLYSCOPE` i.e., 

```
> cd build 
> cmake -DUSE_POLYSCOPE=True ../
``` 
By default, the parameter is set to True. RXMesh implements the necessary functionalities to pass attributes to Polyscope—thanks to its [data adaptors](https://polyscope.run/data_adaptors/). However, this needs attributes to be moved to the host first before passing it to Polyscope. For more information about Polyscope's different visualization options, please checkout Polyscope's [Surface Mesh documentation](https://polyscope.run/structures/surface_mesh/basics/).

  - Example: [render vertex color](./tests/Polyscope_test/test_polyscope.cu)      
    ```cpp
    RXMeshStatic rx("dragon.obj");

    //vertex color attribute 
    auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3);

    //Populate vertex color on the device
    //....
    
    //Move vertex color to the host 
    vertex_color.move(DEVICE, HOST);

    //polyscope instance associated with rx 
    auto polyscope_mesh = rx.get_polyscope_mesh();

    //pass vertex color to polyscope 
    polyscope_mesh->addVertexColorQuantity("vColor", vertex_color);

    //render 
    polyscope::show();
    ```
    <p align="center">
    	<img src="./assets/polyscope_dragon.PNG" width="80%"><br>
    </p>    

### **Matrices and Vectors**
- **Large Matrices:** RXMesh has built-in support for large sparse and dense matrices built on top of [cuSparse](https://docs.nvidia.com/cuda/cusparse/) and [cuBlas](https://docs.nvidia.com/cuda/cublas/), respectively. For example, attributes can be converted to dense matrices as follows 

```cpp

RXMeshStatic rx("input.obj");

//Input mesh coordinates as VertexAttribute
std::shared_ptr<VertexAttribute<float>> x = rx.get_input_vertex_coordinates();

//Convert the attributes to a (#vertices x 3) dense matrix 
std::shared_ptr<DenseMatrix<float>> x_mat = x->to_matrix();

//do something with x_mat
//....

//Populate the VertexAttribute coordinates back with the content of the dense matrix
x->from_matrix(x_mat.get());

```
Dense matrices can be accessed using the usual row and column indices or via the mesh element handle (Vertex/Edge/FaceHandle) as a row index. This allows for easy access to the correct row associated with a specific vertex, edge, or face. Dense matrices support various operations such as absolute sum, AXPY, dot products, norm2, scaling, and swapping.

RXMesh supports sparse matrices, where the sparsity pattern matches the query operations. For example, it is often necessary to build a sparse matrix of size #V x #V with non-zero values at (i, j) only if the vertex corresponding to row i is connected by an edge to the vertex corresponding to column j. Currently, we only support the VV sparsity pattern, but we are working on expanding to all other types of queries.

The sparse matrix can be used to solve a linear system via Cholesky, LU, or QR factorization (relying on [cuSolver](https://docs.nvidia.com/cuda/cusolver/index.html))). The solver offers two APIs. The high-level API reorders the input sparse matrix (to reduce non-zero fill-in after matrix factorization) and allocates the additional memory needed to solve the system. Repeated calls to this API will reorder the matrix and allocate/deallocate the temporary memory with each call. For scenarios where the matrix remains unchanged but multiple right-hand sides need to be solved, users can utilize the low-level API, which splits the solve method into pre_solve() and solve(). The former reorders the matrix and allocates temporary memory only once. The low-level API is currently only supported for Cholesky-based factorization. Check out the MCF application for an example of how to set up and use the solver.

Similar to dense matrices, sparse matrices also support accessing the matrix using the VertexHandle and multiplication by dense matrices.

- **Small Matrices:**
It is often necessary to perform operations on small matrices as part of geometry processing applications, such as computing the SVD of a 3x3 matrix or normalizing a 1x3 vector. For this purpose, RXMesh attributes can be converted into glm or Eigen matrices, as demonstrated in the vertex_normal example above. Both glm and Eigen support small matrix operations inside the GPU kernel.



## **Replicability**
This repo was awarded the [replicability stamp](http://www.replicabilitystamp.org#https-github-com-owensgroup-rxmesh) by the Graphics Replicability Stamp Initiative (GRSI) :tada:. Visit git tag [`v0.1.0`](https://github.com/owensgroup/RXMesh/tree/v0.1.0) for more information about replicability scripts.

## **Bibtex**
```
@article{Mahmoud:2021:RAG,
  author       = {Ahmed H. Mahmoud and Serban D. Porumbescu and John D. Owens},
  title        = {{RXM}esh: A {GPU} Mesh Data Structure},
  journal      = {ACM Transactions on Graphics},
  year         = 2021,
  volume       = 40,
  number       = 4,
  month        = aug,
  issue_date   = {August 2021},
  articleno    = 104,
  numpages     = 16,
  pages        = {104:1--104:16},
  url          = {https://escholarship.org/uc/item/8r5848vp},
  full_talk    = {https://youtu.be/Se_cNAol4hY},
  short_talk   = {https://youtu.be/V_SHMXnCVws},
  doi          = {10.1145/3450626.3459748},
  acmauthorize = {https://dl.acm.org/doi/10.1145/3450626.3459748?cid=81100458295},
  acceptance   = {149/444 (33.6\%)},
  ucdcite      = {a140}
}
```
