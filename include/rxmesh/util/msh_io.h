#pragma once

#include <string>
#include <vector>

#include "rxmesh/types.h"

namespace rxmesh {

MeshKind load_msh(const std::string&                    filename,
                  std::vector<std::vector<rx_coord_t>>& vertices,
                  std::vector<std::vector<uint32_t>>&   simplices,
                  bool                                  append = false);

void save_msh(const std::string&                          filename,
              const std::vector<std::vector<rx_coord_t>>& vertices,
              const std::vector<std::vector<uint32_t>>&   simplices,
              MeshKind                                    kind,
              bool                                        binary = true);

}  // namespace rxmesh
