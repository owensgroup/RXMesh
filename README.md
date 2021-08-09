# **RXMesh** [![Ubuntu](https://github.com/owensgroup/RXMesh/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/owensgroup/RXMesh/actions/workflows/Ubuntu.yml) [![Windows](https://github.com/owensgroup/RXMesh/actions/workflows/Windows.yml/badge.svg)](https://github.com/owensgroup/RXMesh/actions/workflows/Windows.yml)

<p align="center">
    <img src="./assets/david_pacthes.png" width="80%"><br>
</p>

## **About**
RXMesh is a surface triangle mesh data structure and programming model for processing static meshes on the GPU. RXMesh aims at provides a high-performance, generic, and compact data structure that can handle meshes regardless of their quality (e.g., non-manifold). The programming model helps to hide the complexity of the data structure and provides an intuitive access model for different use cases. For more details, please check out our paper:

[RXMesh: A GPU Mesh Data Structure](https://escholarship.org/uc/item/8r5848vp)<br>
[Ahmed H. Mahmoud](https://www.ece.ucdavis.edu/~ahdhn/), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)<br>
[ACM Transaction on Graphics](https://dl.acm.org/doi/abs/10.1145/3450626.3459748) (Proceedings of SIGGRAPH 2021)

This repository provides 1) source code to reproduce the results presented in the paper (git tag `v0.1.0`) and 2) ongoing development of RXMesh. For 1), all input models used in the paper can be found under [TODO]().

## **A Quick Glance**
RXMesh is a CUDA/C++ header-only library. All unit tests are under `tests\` folder. This includes the unit test for some basic functionalities along with the unit test for the query operations. All applications are under `apps\` folder. Please refer to the ReadMe under each application folder for more details. 

## **Compilation**
The code can be compiled on Ubuntu (GCC 9) and Windows (Visual Studio 2019) providing that CUDA (>=11.1.0) is installed. To run the executable(s), an NVIDIA GPU should be installed on the machine.

### Dependencies 
We use the following dependencies:
- [OpenMesh](https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh) to check the applications against reference CPU implementation
- [RapidJson](https://github.com/Tencent/rapidjson) to report the results in JSON file(s)
- [GoogleTest](https://github.com/google/googletest) for unit tests
- [spdlog](https://github.com/gabime/spdlog) for logging

All the dependencies are installed automatically! To compile the code:

```
> git clone https://github.com/owensgroup/RXMesh.git
> cd RXMesh
> mkdir build 
> cd build 
> cmake ../
```
Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system.

## **Bibtex**
```
@article{Mahmoud:2021:RAG,
  author = {Mahmoud, Ahmed H. and Porumbescu, Serban D. and Owens, John D.},
  title = {{RXM}esh: A {GPU} Mesh Data Structure},
  journal = {ACM Transactions on Graphics},
  year = 2021,
  volume = 40,
  number = 4,
  month = aug,
  url = {https://escholarship.org/uc/item/8r5848vp},
  full_talk = {https://youtu.be/Se_cNAol4hY},
  short_talk = {https://youtu.be/V_SHMXnCVws},
  doi = {10.1145/3450626.3459748}
}
```