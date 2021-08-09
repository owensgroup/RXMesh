#include "rxmesh/util/report.h"

class OpenMeshReport : public RXMESH::Report
{
   public:
    OpenMeshReport() : RXMESH::Report()
    {
    }
    OpenMeshReport(const std::string& record_name) : RXMESH::Report(record_name)
    {
    }

    void model_data(const std::string& model_name, const TriMesh& mesh)
    {
        rapidjson::Document subdoc(&this->m_doc.GetAllocator());
        subdoc.SetObject();

        add_member("model_name", model_name, subdoc);
        add_member("num_vertices", static_cast<uint32_t>(mesh.n_vertices()),
                   subdoc);
        add_member("num_edges", static_cast<uint32_t>(mesh.n_edges()), subdoc);
        add_member("num_faces", static_cast<uint32_t>(mesh.n_faces()), subdoc);

        this->m_doc.AddMember("Model", subdoc, m_doc.GetAllocator());
    }
};