#include "core/vector_index_layer.h"
#include "core/vector_index_faiss_ivfflat.h"

namespace lgraph {
VectorIndex::VectorIndex(const std::string& label, const std::string& name, const std::string& distance_type, const std::string& index_type, int vec_dimension, std::vector<int> index_spec)
    : label_(label), name_(name), distance_type_(distance_type), index_type_(index_type), 
      vec_dimension_(vec_dimension), index_spec_(index_spec),
      query_spec_(10),
      vector_index_manager_(size_t(0), label, name) {}

VectorIndex::VectorIndex(const VectorIndex& rhs)
    : label_(rhs.label_), 
      name_(rhs.name_), 
      distance_type_(rhs.distance_type_),
      index_type_(rhs.index_type_),
      vec_dimension_(rhs.vec_dimension_), 
      index_spec_(rhs.index_spec_),
      query_spec_{rhs.query_spec_},
      vector_index_manager_(rhs.vector_index_manager_) {}

bool VectorIndex::SetSearchSpec(int query_spec) {
    query_spec_ = query_spec;
    return true;
}

// factory method
std::unique_ptr<VectorIndex> VectorIndex::Create(const std::string& label, const std::string& name, const std::string& distance_type, const std::string& index_type, int vec_dimension, std::vector<int> index_spec) {
    // TODO
    if (index_type == "IVF_FLAT") {
        int nlist = index_spec.empty() ? 1 : index_spec[0];
        return std::make_unique<FaissIVFFlatIndex>(label, name, distance_type, vec_dimension, nlist);
    }
    return nullptr;
}
}