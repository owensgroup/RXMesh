#pragma once

#include <cuda_runtime_api.h>
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/util.h"
#ifdef __NVCC__
#include "cuda.h"
#include "rxmesh/kernels/get_arch.cuh"
#include "rxmesh/util/cuda_query.h"
#endif
#include "rxmesh/util/git_sha1.h"

#ifdef _WIN32
#include <Winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif

namespace rxmesh {

// Most values are signed and initialized to -1
// if any value is not modified, it won't be written

// To add a test, create a new TestData, fill in the data and use
// add_test to add it to the report
struct TestData
{
    std::vector<float> time_ms;
    int32_t            num_blocks  = -1;
    int32_t            num_threads = -1;
    std::vector<bool>  passed;
    std::string        test_name   = "";
    float              dyn_smem    = -1;
    float              static_smem = -1;
};

struct Report
{
    Report()
    {
    }

    Report(const std::string& record_name)
    {
        m_doc.SetObject();
        m_doc.AddMember("Record Name",
                        rapidjson::Value().SetString(record_name.c_str(),
                                                     record_name.length(),
                                                     m_doc.GetAllocator()),
                        m_doc.GetAllocator());
        std::string str = g_GIT_SHA1;
        m_doc.AddMember("git_sha",
                        rapidjson::Value().SetString(
                            str.c_str(), str.length(), m_doc.GetAllocator()),
                        m_doc.GetAllocator());

        // Time
        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        {
            std::ostringstream oss;
            oss << std::put_time(&tm, "__D%d_%m_%Y__T%H_%M_%S");
            m_output_name_suffix = oss.str() + ".json";
        }

        {
            std::ostringstream oss;
            oss << std::put_time(&tm, "%a %d:%m:%Y %H:%M:%S");
            std::string str = oss.str();

            m_doc.AddMember(
                "date",
                rapidjson::Value().SetString(
                    str.c_str(), str.length(), m_doc.GetAllocator()),
                m_doc.GetAllocator());
        }
    }

    // command line
    void command_line(int argc, char** argv)
    {
        std::string cmd(argv[0]);
        for (int i = 1; i < argc; i++) {
            cmd = cmd + " " + std::string(argv[i]);
        }
        m_doc.AddMember("command_line",
                        rapidjson::Value().SetString(
                            cmd.c_str(), cmd.length(), m_doc.GetAllocator()),
                        m_doc.GetAllocator());
    }

    // GPU
    void device()
    {

        rapidjson::Document subdoc(&m_doc.GetAllocator());
        subdoc.SetObject();

        int device_id = 0;
#ifdef __NVCC__
        CUDA_ERROR(cudaGetDevice(&device_id));
#endif

        cudaDeviceProp devProp;
#ifdef __NVCC__
        CUDA_ERROR(cudaGetDeviceProperties(&devProp, device_id));
#endif

        // ID
        add_member("ID", int(device_id), subdoc);


        // Device Name
        std::string name = devProp.name;
        add_member("Name", name, subdoc);

        // driver
        int ver = 0;
        CUDA_ERROR(cudaDriverGetVersion(&ver));
        add_member("Driver Version", ver, subdoc);

        CUDA_ERROR(cudaRuntimeGetVersion(&ver));
        add_member("Runtime Version", ver, subdoc);
#ifdef __NVCC__
        add_member("CUDA API Version", CUDA_VERSION, subdoc);
#endif
        // Compute Capability
        std::string cc =
            std::to_string(devProp.major) + "." + std::to_string(devProp.minor);
        add_member("Compute Capability", cc, subdoc);

#ifdef __NVCC__
        add_member("__CUDA_ARCH__", cuda_arch(), subdoc);
#endif

        // Memory
        add_member("Total amount of global memory (MB)",
                   (float)devProp.totalGlobalMem / 1048576.0f,
                   subdoc);
        add_member("Total amount of shared memory per block (Kb)",
                   (float)devProp.sharedMemPerBlock / 1024.0f,
                   subdoc);

        // SM
        add_member("Multiprocessors", devProp.multiProcessorCount, subdoc);
#ifdef __NVCC__
        add_member("CUDA Cores/MP",
                   convert_SMV_to_cores(devProp.major, devProp.minor),
                   subdoc);
#endif

        // Clocks
        add_member(
            "GPU Max Clock rate (GHz)", devProp.clockRate * 1e-6f, subdoc);
        add_member(
            "Memory Clock rate (GHz)", devProp.memoryClockRate * 1e-6f, subdoc);
        add_member("Memory Bus Width (bit)", devProp.memoryBusWidth, subdoc);
        add_member("Peak Memory Bandwidth (GB/s)",
                   2.0 * devProp.memoryClockRate *
                       (devProp.memoryBusWidth / 8.0) / 1.0E6,
                   subdoc);

        m_doc.AddMember("GPU Device", subdoc, m_doc.GetAllocator());
    }

    // System
    void system()
    {
        rapidjson::Document subdoc(&m_doc.GetAllocator());
        subdoc.SetObject();
#ifdef _WIN32
        // https://stackoverflow.com/a/11828223/1608232
        char    szPath[128] = "";
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        gethostname(szPath, sizeof(szPath));
        std::string hostname(szPath);
        add_member("Hostname", hostname, subdoc);
        WSACleanup();
#else
        char hostname[300];
        gethostname(hostname, 300 - 1);
        hostname[300 - 1] = '\0';
        std::string hostname_str(hostname);
        add_member("Hostname", hostname_str, subdoc);
#endif


#ifdef _MSC_VER
        add_member(
            "Microsoft Full Compiler Version", int32_t(_MSC_FULL_VER), subdoc);
        add_member("Microsoft Compiler Version", int32_t(_MSC_VER), subdoc);
#else

        // https://stackoverflow.com/a/38531037/1608232
        std::string true_cxx =
#ifdef __clang__
            "clang++";
#elif __GNUC__
            "g++";
#else
            "unknown"
#endif
        auto ver_string = [](int a, int b, int c) {
            std::ostringstream ss;
            ss << a << '.' << b << '.' << c;
            return ss.str();
        };

        std::string true_cxx_ver =
#ifdef __clang__
            ver_string(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif __GNUC__
            ver_string(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
            "-1";
#endif

        add_member("compiler_name", true_cxx, subdoc);
        add_member("compiler_version", true_cxx_ver, subdoc);
#endif
        add_member("C++ version", int(__cplusplus), subdoc);

#if NDEBUG
        std::string build_mode = "Release";
        add_member("Build Mode", build_mode, subdoc);
#else
        std::string build_mode = "Debug";
        add_member("Build Mode", build_mode, subdoc);

#endif
        m_doc.AddMember("System", subdoc, m_doc.GetAllocator());
    }

    // write report to file
    // file will be located in OUTPUT_DIR
    void write(const std::string& output_folder,
               const std::string& output_filename,
               bool               append_time_to_file_name = true)
    {
        std::string full_name =
            output_folder + "/" + remove_extension(output_filename) +
            (append_time_to_file_name ? m_output_name_suffix : ".json");

        // create the folder if it does not exist
        if (!std::filesystem::is_directory(output_folder) ||
            !std::filesystem::exists(output_folder)) {
            std::filesystem::create_directories(output_folder);
        }

        std::ofstream ofs(full_name);
        if (!ofs.is_open()) {
            RXMESH_ERROR("Report::write() can not open {}", full_name);
        }
        rapidjson::OStreamWrapper                          osw(ofs);
        rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
        m_doc.Accept(writer);
    }

    // get model data from RXMesh
    void model_data(const std::string& model_name, const rxmesh::RXMesh& rxmesh)
    {
        rapidjson::Document subdoc(&m_doc.GetAllocator());
        subdoc.SetObject();

        add_member("model_name", model_name, subdoc);
        add_member("num_vertices", rxmesh.get_num_vertices(), subdoc);
        add_member("num_edges", rxmesh.get_num_edges(), subdoc);
        add_member("num_faces", rxmesh.get_num_faces(), subdoc);
        add_member("max_valence", rxmesh.get_max_valence(), subdoc);
        add_member("is_edge_manifold", rxmesh.is_edge_manifold(), subdoc);
        add_member("is_closed", rxmesh.is_closed(), subdoc);
        add_member("patch_size", rxmesh.get_patch_size(), subdoc);
        add_member("num_patches", rxmesh.get_num_patches(), subdoc);
        add_member("num_components", rxmesh.get_num_components(), subdoc);
        add_member("num_lloyd_run", rxmesh.get_num_lloyd_run(), subdoc);
        add_member("patching_time", rxmesh.get_patching_time(), subdoc);
        uint32_t min_patch_size(0), max_patch_size(0), avg_patch_size(0);
        rxmesh.get_max_min_avg_patch_size(
            min_patch_size, max_patch_size, avg_patch_size);
        add_member("min_patch_size", min_patch_size, subdoc);
        add_member("max_patch_size", max_patch_size, subdoc);
        add_member("avg_patch_size", avg_patch_size, subdoc);
        add_member("per_patch_max_vertices",
                   rxmesh.get_per_patch_max_vertices(),
                   subdoc);
        add_member(
            "per_patch_max_edges", rxmesh.get_per_patch_max_edges(), subdoc);
        add_member(
            "per_patch_max_faces", rxmesh.get_per_patch_max_faces(), subdoc);
        add_member("ribbon_overhead (%)", rxmesh.get_ribbon_overhead(), subdoc);
        add_member(
            "total_gpu_storage (mb)", rxmesh.get_gpu_storage_mb(), subdoc);
        m_doc.AddMember("Model", subdoc, m_doc.GetAllocator());
    }

    // add test using TestData
    void add_test(const TestData& test_data)
    {
        rapidjson::Document subdoc(&m_doc.GetAllocator());
        subdoc.SetObject();

        if (test_data.num_blocks != -1) {
            add_member("num_blocks", test_data.num_blocks, subdoc);
        }

        if (test_data.num_threads != -1) {
            add_member("num_threads", test_data.num_threads, subdoc);
        }

        if (test_data.dyn_smem != -1) {
            add_member("dynamic_shared_memory (b)", test_data.dyn_smem, subdoc);
        }

        if (test_data.static_smem != -1) {
            add_member(
                "static_shared_memory (b)", test_data.static_smem, subdoc);
        }

        if (!test_data.passed.empty()) {
            add_member("passed", test_data.passed, subdoc);
        }

        if (!test_data.time_ms.empty()) {
            add_member("time (ms)", test_data.time_ms, subdoc);
        }

        rapidjson::Value key(test_data.test_name.c_str(),
                             subdoc.GetAllocator());

        m_doc.AddMember(key, subdoc, m_doc.GetAllocator());
    }

    // add members to the main object
    template <typename T>
    void add_member(std::string member_key, const T member_val)
    {
        add_member(member_key, member_val, m_doc);
    }

    //////////////////////////////////////////////////////////////////////////


    rapidjson::Document m_doc;

   protected:
    std::string m_output_name_suffix;

    template <typename docT>
    void add_member(std::string member_key, const int32_t member_val, docT& doc)
    {
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());
        doc.AddMember(
            key, rapidjson::Value().SetInt(member_val), doc.GetAllocator());
    }
    template <typename docT>
    void add_member(std::string    member_key,
                    const uint32_t member_val,
                    docT&          doc)
    {
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());
        doc.AddMember(
            key, rapidjson::Value().SetUint(member_val), doc.GetAllocator());
    }

    template <typename docT>
    void add_member(std::string member_key, const double member_val, docT& doc)
    {
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());
        doc.AddMember(
            key, rapidjson::Value().SetDouble(member_val), doc.GetAllocator());
    }

    template <typename docT>
    void add_member(std::string member_key, const bool member_val, docT& doc)
    {
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());
        doc.AddMember(
            key, rapidjson::Value().SetBool(member_val), doc.GetAllocator());
    }

    template <typename docT>
    void add_member(std::string       member_key,
                    const std::string member_val,
                    docT&             doc)
    {
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());
        doc.AddMember(
            key,
            rapidjson::Value().SetString(
                member_val.c_str(), member_val.length(), doc.GetAllocator()),
            doc.GetAllocator());
    }

    template <typename T, typename docT>
    void add_member(std::string           member_key,
                    const std::vector<T>& member_val,
                    docT&                 doc)
    {
        rapidjson::Value val(rapidjson::kArrayType);
        rapidjson::Value key(member_key.c_str(), doc.GetAllocator());

        for (size_t i = 0; i < member_val.size(); ++i) {
            val.PushBack(rapidjson::Value(member_val[i]).Move(),
                         doc.GetAllocator());
        }
        doc.AddMember(key, val, doc.GetAllocator());
    }
};


class CustomReport : public Report
{
   public:
    CustomReport() : Report()
    {
    }
    CustomReport(const std::string& record_name) : Report(record_name)
    {
    }

    void model_data(const std::string& model_name,
                    const uint32_t     num_vertices,
                    const uint32_t     num_faces)
    {
        rapidjson::Document subdoc(&this->m_doc.GetAllocator());
        subdoc.SetObject();

        add_member("model_name", model_name, subdoc);
        add_member("num_vertices", num_vertices, subdoc);
        add_member("num_faces", num_faces, subdoc);

        this->m_doc.AddMember("Model", subdoc, m_doc.GetAllocator());
    }
};
}  // namespace rxmesh