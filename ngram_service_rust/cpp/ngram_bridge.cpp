#include "ngram.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CorpusHandle {
  explicit CorpusHandle(size_t capacity, const ngram::Param& param)
      : corpus(std::make_unique<ngram::Ngram>(capacity, param)), draft_token_num(param.draft_token_num) {}

  std::unique_ptr<ngram::Ngram> corpus;
  size_t draft_token_num;
};

thread_local std::string last_error;

void set_error(const std::exception& error) {
  last_error = error.what();
}

void validate_offsets(const int64_t* offsets, size_t batch_size, size_t token_count) {
  if (offsets == nullptr || offsets[0] != 0 || offsets[batch_size] != static_cast<int64_t>(token_count)) {
    throw std::runtime_error("invalid CSR offsets");
  }
  for (size_t i = 0; i < batch_size; ++i) {
    if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) {
      throw std::runtime_error("CSR offsets must be nonnegative and nondecreasing");
    }
  }
}

}  // namespace

extern "C" {

const char* ngram_last_error() {
  return last_error.c_str();
}

void* ngram_create(
    uint64_t capacity,
    uint64_t max_trie_depth,
    uint64_t min_bfs_breadth,
    uint64_t max_bfs_breadth,
    uint64_t draft_token_num,
    int match_type) {
  try {
    ngram::Param param;
    param.enable = true;
    param.enable_router_mode = false;
    param.max_trie_depth = static_cast<size_t>(max_trie_depth);
    param.min_bfs_breadth = static_cast<size_t>(min_bfs_breadth);
    param.max_bfs_breadth = static_cast<size_t>(max_bfs_breadth);
    param.draft_token_num = static_cast<size_t>(draft_token_num);
    param.match_type = match_type == 0 ? "BFS" : "PROB";
    return new CorpusHandle(static_cast<size_t>(capacity), param);
  } catch (const std::exception& error) {
    set_error(error);
    return nullptr;
  }
}

void ngram_destroy(void* raw_handle) {
  delete static_cast<CorpusHandle*>(raw_handle);
}

int ngram_batch_put(
    void* raw_handle,
    const int32_t* flat_tokens,
    size_t token_count,
    const int64_t* offsets,
    size_t batch_size,
    int wait_for_visibility) {
  try {
    if (raw_handle == nullptr || flat_tokens == nullptr || batch_size == 0) {
      throw std::runtime_error("invalid batch_put arguments");
    }
    validate_offsets(offsets, batch_size, token_count);
    std::vector<std::vector<int32_t>> rows(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      rows[i].assign(flat_tokens + offsets[i], flat_tokens + offsets[i + 1]);
    }
    auto* handle = static_cast<CorpusHandle*>(raw_handle);
    handle->corpus->asyncInsert(std::move(rows));
    if (wait_for_visibility != 0) {
      handle->corpus->synchronize();
    }
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return -1;
  }
}

int ngram_batch_get(
    void* raw_handle,
    const int64_t* state_ids,
    const int64_t* total_lens,
    const int32_t* flat_tokens,
    size_t token_count,
    const int64_t* offsets,
    size_t batch_size,
    int32_t* output_tokens,
    uint8_t* output_mask) {
  try {
    if (raw_handle == nullptr || state_ids == nullptr || total_lens == nullptr || flat_tokens == nullptr ||
        output_tokens == nullptr || output_mask == nullptr || batch_size == 0) {
      throw std::runtime_error("invalid batch_get arguments");
    }
    validate_offsets(offsets, batch_size, token_count);

    std::vector<int64_t> states(state_ids, state_ids + batch_size);
    std::vector<size_t> lengths(batch_size);
    std::vector<std::vector<int32_t>> rows(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      if (total_lens[i] <= 0) {
        throw std::runtime_error("total_lens must be positive");
      }
      lengths[i] = static_cast<size_t>(total_lens[i]);
      rows[i].assign(flat_tokens + offsets[i], flat_tokens + offsets[i + 1]);
    }

    auto* handle = static_cast<CorpusHandle*>(raw_handle);
    auto result = handle->corpus->batchMatch(states, rows, lengths);
    const size_t token_size = batch_size * handle->draft_token_num;
    const size_t mask_size = token_size * handle->draft_token_num;
    if (result.token.size() != token_size || result.mask.size() != mask_size) {
      throw std::runtime_error("unexpected NGRAM result shape");
    }
    std::memcpy(output_tokens, result.token.data(), token_size * sizeof(int32_t));
    std::memcpy(output_mask, result.mask.data(), mask_size * sizeof(uint8_t));
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return -1;
  }
}

int ngram_erase_match_state(void* raw_handle, const int64_t* state_ids, size_t count) {
  try {
    if (raw_handle == nullptr || (count != 0 && state_ids == nullptr)) {
      throw std::runtime_error("invalid erase arguments");
    }
    std::vector<int64_t> states(state_ids, state_ids + count);
    static_cast<CorpusHandle*>(raw_handle)->corpus->eraseMatchState(states);
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return -1;
  }
}

}  // extern "C"
