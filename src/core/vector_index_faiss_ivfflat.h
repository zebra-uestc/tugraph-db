#pragma once

#include <vector>
#include <cstdint>
#include "core/vector_index_layer.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "faiss/impl/io.h"

namespace lgraph {

// IVFFlat index
class FaissIVFFlatIndex : public VectorIndex {
 private:
    faiss::IndexFlatL2* quantizer_;
    faiss::IndexIVFFlat* index_;
    int nlist_;

 public:
    FaissIVFFlatIndex(const std::string& label, const std::string& name, const std::string& distance_type, int vec_dimension, int nlist);

    ~FaissIVFFlatIndex() override;

    bool Add(const std::vector<std::vector<float>>& vectors, size_t num_vectors) override;

    bool Build() override;

    std::vector<uint8_t> Save() override;

    void Load(std::vector<uint8_t>& idx_bytes) override;

    bool Search(const std::vector<float>& query, size_t num_results, std::vector<float>& distances, std::vector<int64_t>& indices) override;

    std::unique_ptr<VectorIndex> Clone() const override;
};
}  // namespace lgraph