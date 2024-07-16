#pragma once

#include <vector>
#include <cstdint>
#include "core/vector_index_manager.h"

namespace lgraph {

class VectorIndex {
   friend class Schema;
   friend class LightningGraph;
   friend class Transaction;
   friend class IndexManager;

 protected:
   std::string label_;
   std::string name_;
   std::string distance_type_;
   std::string index_type_;
   int vec_dimension_;
   std::vector<int> index_spec_;
   int query_spec_;
   VectorIndexManager vector_index_manager_; 

 public:
    VectorIndex(const std::string& label, const std::string& name, const std::string& distance_type, const std::string& index_type, int vec_dimension, std::vector<int> index_spec);

    VectorIndex(const VectorIndex& rhs);

    VectorIndex(VectorIndex&& rhs) = delete;

    VectorIndex& operator=(const VectorIndex& rhs) = delete;

    VectorIndex& operator=(VectorIndex&& rhs) = delete;

    // get label
    std::string GetLabel() { return label_; }

    // get name
    std::string GetName() { return name_; }

    //get distance type
    std::string GetDistanceType() { return distance_type_; }

    //get index type
    std::string GetIndexType() { return index_type_; }

    //get vector dimension
    int GetVecDimension() { return vec_dimension_; }

    // get vector_index_manager
    VectorIndexManager* GetVectorIndexManager() { return &vector_index_manager_; }

    //set search specification
    bool SetSearchSpec(int query_spec);

    // add vector to index and build index
    virtual bool Add(const std::vector<std::vector<float>>& vectors, size_t num_vectors) = 0;

    // build index
    virtual bool Build() = 0;

    // serialize index
    virtual std::vector<uint8_t> Save() = 0;

    // load index form serialization
    virtual void Load(std::vector<uint8_t>& idx_bytes) = 0;

    // search vector in index
    virtual bool Search(const std::vector<float>& query, size_t num_results, std::vector<float>& distances, std::vector<int64_t>& indices) = 0;

    virtual ~VectorIndex() = default;

    // factory method
    static std::unique_ptr<VectorIndex> Create(const std::string& label, const std::string& name, const std::string& distance_type, const std::string& index_type, int vec_dimension, std::vector<int> index_spec);

    // clone method
    virtual std::unique_ptr<VectorIndex> Clone() const = 0;
};
}  // namespace lgraph