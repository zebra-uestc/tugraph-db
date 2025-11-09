/**
 * Copyright 2022 AntGroup CO., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <utility>
#include "core/vsag_ivf_flat.h"
#include "tools/lgraph_log.h"
#include "fma-common/string_formatter.h"
#include "lgraph/lgraph_exceptions.h"

namespace lgraph {
VsagIVFFlat::VsagIVFFlat(const std::string& label, const std::string& name,
                            const std::string& distance_type,
                            const std::string& index_type, int vec_dimension,
                            std::vector<int> index_spec)
    : VectorIndex(label, name, distance_type, index_type,
                     vec_dimension, std::move(index_spec)) {
    Build();
    LOG_INFO() << FMA_FMT("Create IVF_Flat instance, {}:{}", GetLabel(), GetName());
}
 
VsagIVFFlat::~VsagIVFFlat() {
    LOG_INFO() << FMA_FMT("Destroy IVF_Flat instance, {}:{}", GetLabel(), GetName());
    index_ = nullptr;
}

// add vector to index
void VsagIVFFlat::Add(const std::vector<std::vector<float>>& vectors,
                      const std::vector<int64_t>& vids) {
    if (vectors.size() != vids.size()) {
        THROW_CODE(VectorIndexException,
                   "size mismatch, vectors.size:{}, vids.size:{}", vectors.size(), vids.size());
    }
    if (vectors.empty()) return;

    auto num_vectors = vectors.size();
    auto* index_vectors = new float[num_vectors * vec_dimension_];
    auto* ids = new int64_t[num_vectors];

    for (size_t i = 0; i < num_vectors; i++) {
        std::copy(vectors[i].begin(), vectors[i].end(), &index_vectors[i * vec_dimension_]);
    }
    for (size_t i = 0; i < num_vectors; i++) {
        vectorid_++;
        ids[i] = vectorid_;
        if (vid_vectorid_.count(vids[i])) {
            delete[] ids;
            delete[] index_vectors;
            THROW_CODE(VectorIndexException, "[VsagIVFFlat Add] vid {} already exists", vids[i]);
        }
        vid_vectorid_[vids[i]] = vectorid_;
        vectorid_vid_[vectorid_] = {false, vids[i]};
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(vec_dimension_)
           ->NumElements(num_vectors)
           ->Ids(ids)
           ->Float32Vectors(index_vectors);
    if (index_->GetNumElements() == 0) {
        auto build_result = index_->Build(dataset);
        if (build_result.has_value()) {
        } else {
            delete[] ids;
            delete[] index_vectors;
            THROW_CODE(VectorIndexException, build_result.error().message);
        }
    } else {
        auto result = index_->Add(dataset);
        if (result.has_value()) {
            if (!result.value().empty()) {
                delete[] ids;
                delete[] index_vectors;
                THROW_CODE(VectorIndexException,
                           "add vector into index, {} failed", result.value().size());
            }
        } else {
            delete[] ids;
            delete[] index_vectors;
            THROW_CODE(VectorIndexException, "add vector into index, error:{}", result.error().message);
        }
    }
}
 
void VsagIVFFlat::Clear() {
    vectorid_ = 0;
    index_ = nullptr;
    deleted_vectorid_ = 0;
    std::unordered_map<int64_t, int64_t>().swap(vid_vectorid_);
    std::unordered_map<int64_t, std::pair<bool, int64_t>>().swap(vectorid_vid_);
    Build();
}

void VsagIVFFlat::Remove(const std::vector<int64_t>& vids) {
    for (auto vid : vids) {
        auto iter = vid_vectorid_.find(vid);
        if (iter == vid_vectorid_.end()) {
            THROW_CODE(VectorIndexException, "[VsagIVFFlat Remove] vid {} does not exist", vid);
        }
        vectorid_vid_.at(iter->second) = {true, -1};
        deleted_vectorid_++;
        vid_vectorid_.erase(iter);
    }
}
 
void VsagIVFFlat::Build() {
    nlohmann::json index_param = {
        {"buckets_count", index_spec_[0]},
        {"base_quantization_type", "fp32"},
        {"partition_strategy_type", "ivf"},
        {"ivf_train_type", "kmeans"}
    };
    nlohmann::json index_parameters = {
        {"dtype", "float32"},
        {"metric_type", distance_type_},
        {"dim", vec_dimension_},
        {"index_param", index_param}
    };
    auto temp = vsag::Factory::CreateIndex("ivf", index_parameters.dump());
    if (temp.has_value()) {
        index_ = std::move(temp.value());
    } else {
        THROW_CODE(VectorIndexException, temp.error().message);
    }
}
 
// serialize index
std::vector<uint8_t> VsagIVFFlat::Save() {
    std::vector<uint8_t> blob;
    if (!index_) {
        return blob;
    }
    if (auto bs = index_->Serialize(); bs.has_value()) {
        std::vector<uint8_t> mapping_blob;
        {
            uint64_t count = vectorid_vid_.size();
            mapping_blob.resize(sizeof(uint64_t) + count * (sizeof(int64_t) * 2));
            char *p = reinterpret_cast<char*>(mapping_blob.data());
            // write count
            std::memcpy(p, &count, sizeof(uint64_t));
            p += sizeof(uint64_t);

            for (const auto &kv : vectorid_vid_) {
                int64_t vectorid = kv.first;
                int64_t user_vid = kv.second.second;
                std::memcpy(p, &vectorid, sizeof(int64_t));
                p += sizeof(int64_t);
                std::memcpy(p, &user_vid, sizeof(int64_t));
                p += sizeof(int64_t);
            }
        }

        auto keys = bs->GetKeys();
        std::vector<std::string> keys2 = keys;
        const std::string mapping_key = "lgraph.wrapper.mapping";
        keys2.push_back(mapping_key);

        std::ofstream file("ivf_flat.index", std::ios::binary);
        std::vector<uint64_t> offsets;
        uint64_t offset = 0;
        // write normal keys' binaries first
        for (const auto &key : keys) {
            vsag::Binary b = bs->Get(key);
            writeBinaryPOD(file, b.size);
            file.write(reinterpret_cast<const char*>(b.data.get()), b.size);
            offsets.push_back(offset);
            offset += sizeof(b.size) + b.size;
        }
        {
            uint64_t msize = mapping_blob.size();
            writeBinaryPOD(file, msize);
            if (msize) file.write(reinterpret_cast<const char*>(mapping_blob.data()), msize);
            offsets.push_back(offset);
            offset += sizeof(msize) + msize;
        }
        for (uint64_t i = 0; i < keys2.size(); ++i) {
            const auto &key = keys2[i];
            int64_t len = static_cast<int64_t>(key.length());
            writeBinaryPOD(file, len);
            file.write(key.c_str(), len);
            writeBinaryPOD(file, offsets[i]);
        }
        writeBinaryPOD(file, keys2.size());
        writeBinaryPOD(file, offset);
        file.close();

        std::ifstream input_file("ivf_flat.index", std::ios::binary | std::ios::ate);
        if (input_file.is_open()) {
            std::streamsize size = input_file.tellg();
            input_file.seekg(0, std::ios::beg);
            blob.resize(size);
            input_file.read(reinterpret_cast<char*>(blob.data()), size);
            input_file.close();
        }
    }
    std::remove("ivf_flat.index");
    return blob;
}
 
// load index form serialization
void VsagIVFFlat::Load(std::vector<uint8_t>& idx_bytes) {
    const std::string filename = "ivf_flat.index";
    std::ofstream output_file(filename, std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(idx_bytes.data()), idx_bytes.size());
    output_file.close();
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    file.seekg(-static_cast<int>(sizeof(uint64_t) * 2), std::ios::end);
    if (file.fail()) {
        file.close();
        return;
    }
    uint64_t num_keys = 0, footer_offset = 0;
    readBinaryPOD(file, num_keys);
    readBinaryPOD(file, footer_offset);
    if (num_keys == 0 || footer_offset == 0) {
        file.close();
        return;
    }
    file.seekg(footer_offset, std::ios::beg);
    std::vector<std::string> keys;
    std::vector<uint64_t> offsets;
    for (uint64_t i = 0; i < num_keys; ++i) {
        int64_t key_len = 0;
        readBinaryPOD(file, key_len);
        std::vector<char> key_buf(key_len);
        file.read(key_buf.data(), key_len);
        keys.push_back(std::string(key_buf.begin(), key_buf.end()));
        uint64_t offset = 0;
        readBinaryPOD(file, offset);
        offsets.push_back(offset);
    }

    vsag::ReaderSet bs;
    for (uint64_t i = 0; i < num_keys; ++i) {
        int64_t size = (i + 1 == num_keys) ? (footer_offset - offsets[i] - sizeof(uint64_t))
                                           : (offsets[i + 1] - offsets[i] - sizeof(uint64_t));
        auto file_reader = vsag::Factory::CreateLocalFileReader(filename,
                                                                offsets[i] + sizeof(uint64_t), size);
        bs.Set(keys[i], file_reader);
    }

    file.close();
    index_->Deserialize(bs);

    std::ifstream file2(filename, std::ios::in | std::ios::binary);
    if (!file2.is_open()) {
        return;
    }

    const std::string mapping_key = "lgraph.wrapper.mapping";
    for (uint64_t i = 0; i < num_keys; ++i) {
        if (keys[i] == mapping_key) {
            int64_t size = (i + 1 == num_keys) ? (footer_offset - offsets[i] - sizeof(uint64_t))
                                               : (offsets[i + 1] - offsets[i] - sizeof(uint64_t));
            uint64_t start = offsets[i] + sizeof(uint64_t);
            file2.seekg(start, std::ios::beg);
            std::vector<char> buf(size);
            file2.read(buf.data(), size);
            const char *p = buf.data();
            uint64_t count = 0;
            std::memcpy(&count, p, sizeof(uint64_t));
            p += sizeof(uint64_t);
            vectorid_vid_.clear();
            vid_vectorid_.clear();
            int64_t max_vectorid = 0;
            for (uint64_t k = 0; k < count; ++k) {
                int64_t vectorid = 0, user_vid = 0;
                std::memcpy(&vectorid, p, sizeof(int64_t)); p += sizeof(int64_t);
                std::memcpy(&user_vid, p, sizeof(int64_t)); p += sizeof(int64_t);
                vectorid_vid_[vectorid] = {false, user_vid};
                vid_vectorid_[user_vid] = vectorid;
                if (vectorid > max_vectorid) max_vectorid = vectorid;
            }
            vectorid_ = max_vectorid; // restore counter
            break;
        }
    }
    file2.close();
    std::remove(filename.c_str());
}
 
// search vector in index
std::vector<std::pair<int64_t, float>>
VsagIVFFlat::KnnSearch(const std::vector<float>& query, int64_t top_k, int scan_buckets) {
    auto* query_copy = new float[query.size()];
    std::copy(query.begin(), query.end(), query_copy);
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(1)->Dim(vec_dimension_)->Float32Vectors(query_copy)->Owner(true);
    nlohmann::json parameters {
        {"ivf", {{"scan_buckets_count", scan_buckets}}}
    };
    std::vector<std::pair<int64_t, float>> ret;
    auto result = index_->KnnSearch(dataset, top_k, parameters.dump(),
        [this](int64_t id)->bool { return vectorid_vid_.at(id).first; });
    if (result.has_value()) {
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            auto vector_id = result.value()->GetIds()[i];
            ret.emplace_back(vectorid_vid_.at(vector_id).second, result.value()->GetDistances()[i]);
        }
    } else {
        THROW_CODE(VectorIndexException, result.error().message);
    }
    return ret;
}

std::vector<std::pair<int64_t, float>>
VsagIVFFlat::RangeSearch(const std::vector<float>& query, float radius, int ef_search, int limit) {
    if (query.empty()) {
        THROW_CODE(InputError, "please check the input");
    }
    if (!index_) {
        THROW_CODE(InputError, "index not initialized");
    }
    nlohmann::json parameters{
        {"ivf", {{"scan_buckets_count", ef_search}}}
    };

    auto* query_copy = new float[query.size()];
    std::copy(query.begin(), query.end(), query_copy);
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(1)->Dim(vec_dimension_)->Float32Vectors(query_copy);

    std::vector<std::pair<int64_t, float>> ret;
    auto result = index_->RangeSearch(dataset, radius, parameters.dump(),
        [this](int64_t id)->bool {
            return vectorid_vid_.at(id).first;
        }, limit);

    if (result.has_value()) {
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            int64_t vector_id = result.value()->GetIds()[i];
            ret.emplace_back(vectorid_vid_.at(vector_id).second, result.value()->GetDistances()[i]);
        }
    } else {
        THROW_CODE(VectorIndexException, result.error().message);
    }
    return ret;
}

int64_t VsagIVFFlat::GetElementsNum() {
    return index_->GetNumElements();
}
 
int64_t VsagIVFFlat::GetMemoryUsage() {
    return index_->GetMemoryUsage();
}
 
int64_t VsagIVFFlat::GetDeletedIdsNum() {
    return deleted_vectorid_;
}
 
}  // namespace lgraph
