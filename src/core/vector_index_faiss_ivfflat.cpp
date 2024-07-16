#include "core/vector_index_faiss_ivfflat.h"

namespace lgraph {
FaissIVFFlatIndex::FaissIVFFlatIndex(const std::string& label, const std::string& name, const std::string& distance_type, int vec_dimension, int nlist)
    : VectorIndex(label, name, distance_type, "IVF_FLAT", vec_dimension, {nlist}), 
      quantizer_(nullptr), 
      index_(nullptr),
      nlist_(nlist) {}

FaissIVFFlatIndex::~FaissIVFFlatIndex() {
    delete quantizer_;
    delete index_;
}

// add vector to index and build index
bool FaissIVFFlatIndex::Add(const std::vector<std::vector<float>>& vectors, size_t num_vectors) {
    if (!quantizer_) {
        quantizer_ = new faiss::IndexFlatL2(vec_dimension_);
    }
    if (!index_) {
        index_ = new faiss::IndexIVFFlat(quantizer_, vec_dimension_, nlist_);
    }
    //reduce dimension
    std::vector<float> index_vectors;
    index_vectors.reserve(num_vectors * vec_dimension_);
    for (const auto& vec : vectors) {
        index_vectors.insert(index_vectors.end(), vec.begin(), vec.end());
    }
    // train after build quantizer
    assert(!index_->is_trained);
    index_->train(num_vectors, index_vectors.data());
    assert(index_->is_trained);
    index_->add(num_vectors, index_vectors.data());
    return true;
}

// build index
bool FaissIVFFlatIndex::Build() {
    if (!quantizer_) {
        quantizer_ = new faiss::IndexFlatL2(vec_dimension_);
    }
    if (!index_) {
        index_ = new faiss::IndexIVFFlat(quantizer_, vec_dimension_, nlist_);
    }
    return true;
}

// serialize index
std::vector<uint8_t> FaissIVFFlatIndex::Save() {
    faiss::VectorIOWriter writer;
    faiss::write_index(index_, &writer, 0);
    return writer.data;
}

// load index form serialization
void FaissIVFFlatIndex::Load(std::vector<uint8_t>& idx_bytes) {
    faiss::VectorIOReader reader;
    reader.data = idx_bytes;
    index_ = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(&reader));
}

// search vector in index
bool FaissIVFFlatIndex::Search(const std::vector<float>& query, size_t num_results, std::vector<float>& distances, std::vector<int64_t>& indices) {
    if (query.empty() || num_results == 0) {
        return false;
    }
    distances.resize(num_results * 1);
    indices.resize(num_results * 1);
    index_->nprobe = static_cast<size_t>(query_spec_);
    index_->search(1, query.data(), num_results, distances.data(), indices.data());
    return !indices.empty();
}

std::unique_ptr<VectorIndex> FaissIVFFlatIndex::Clone() const {
    return std::make_unique<FaissIVFFlatIndex>(*this);
}
}