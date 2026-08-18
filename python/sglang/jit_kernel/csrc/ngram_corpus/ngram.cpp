#include "ngram.h"

#include "trie.h"
#include <limits>
#include <stdexcept>
#include <string>

namespace ngram {

Ngram::Ngram(size_t capacity, const Param& param) : param_(param) {
  if (!(param_.max_trie_depth > 1)) {
    throw std::runtime_error(
        "param_.max_trie_depth must be greater than 1, current value: " + std::to_string(param_.max_trie_depth));
  }
  if (!(param_.min_bfs_breadth > 0)) {
    throw std::runtime_error(
        "min_bfs_breadth must be greater than 0, current value: " + std::to_string(param_.min_bfs_breadth));
  }
  if (!(param_.min_bfs_breadth <= param_.max_bfs_breadth)) {
    throw std::runtime_error(
        "min_bfs_breadth must be less than or equal to max_bfs_breadth, "
        "current min_bfs_breadth: " +
        std::to_string(param_.min_bfs_breadth) + ", max_bfs_breadth: " + std::to_string(param_.max_bfs_breadth));
  }
  if (!(param_.draft_token_num > 0)) {
    throw std::runtime_error(
        "draft_token_num must be greater than 0, current value: " + std::to_string(param_.draft_token_num));
  }
  for (auto config : param_.batch_draft_token_num) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config <= param_.draft_token_num)) {
        throw std::runtime_error(
            "batch_draft_token_num config value " + std::to_string(config) +
            " must be less than or equal to draft_token_num: " + std::to_string(param_.draft_token_num));
      }
    }
  }

  trie_ = std::make_unique<Trie>(capacity, param_);

  insert_worker_ = std::thread(&Ngram::insertWorker, this);
}

Ngram::~Ngram() {
  insert_queue_.close();
  if (insert_worker_.joinable()) {
    insert_worker_.join();
  }
}

void Ngram::synchronize() const {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  sync_cv_.wait(lock, [this] { return pending_count_ == 0; });
}

void Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_count_ += tokens.size();
  }
  for (auto&& token : tokens) {
    insert_queue_.enqueue(std::move(token));
  }
}

// NOTE: staging operations (start/append/finish) are called from a background
// thread during async corpus loading. They do not take the shared Trie lock because
// staging_sam_ is disjoint from sams_ / trie_. Only finishExternalCorpusLoad
// briefly acquires trie_mutex_ when moving the completed SAM into sams_.
void Ngram::startExternalCorpusLoad() {
  if (staging_sam_) {
    throw std::runtime_error("startExternalCorpusLoad called while another load is in progress");
  }
  staging_sam_ = std::make_unique<SuffixAutomaton>();
}

void Ngram::appendExternalCorpusTokens(const std::vector<int32_t>& tokens) {
  if (!staging_sam_) {
    throw std::runtime_error("appendExternalCorpusTokens called without startExternalCorpusLoad");
  }
  staging_sam_->appendTokens(tokens);
}

void Ngram::finishExternalCorpusLoad(const std::string& corpus_id) {
  if (!staging_sam_) {
    throw std::runtime_error("finishExternalCorpusLoad called without startExternalCorpusLoad");
  }
  staging_sam_->finalize();
  if (staging_sam_->empty()) {
    staging_sam_.reset();
    throw std::runtime_error("External corpus is empty — no tokens were loaded.");
  }
  // Only lock briefly to install the completed SAM.
  std::unique_lock<std::shared_mutex> lock(trie_mutex_);
  if (sams_.find(corpus_id) != sams_.end()) {
    throw std::runtime_error(
        "External corpus '" + corpus_id + "' already exists. Remove it before adding a new corpus with the same id.");
  }
  sams_.emplace(corpus_id, std::move(staging_sam_));
}

void Ngram::removeExternalCorpus(const std::string& corpus_id) {
  std::unique_lock<std::shared_mutex> lock(trie_mutex_);
  sams_.erase(corpus_id);
}

void Ngram::resetStagingSam() {
  // staging_sam_ is only accessed from the loading thread — no lock needed.
  staging_sam_.reset();
}

void Ngram::clearExternalCorpus() {
  std::unique_lock<std::shared_mutex> lock(trie_mutex_);
  sams_.clear();
  staging_sam_.reset();
}

std::vector<std::pair<std::string, int64_t>> Ngram::listExternalCorpora() const {
  std::shared_lock<std::shared_mutex> lock(trie_mutex_);
  std::vector<std::pair<std::string, int64_t>> entries;
  entries.reserve(sams_.size());
  for (const auto& [id, sam] : sams_) {
    entries.emplace_back(id, sam->tokenCount());
  }
  return entries;
}

void Ngram::insertWorker() {
  for (;;) {
    std::vector<int32_t> data;
    if (!insert_queue_.dequeue(data)) {
      break;
    }
    {
      std::unique_lock<std::shared_mutex> lock(trie_mutex_);
      trie_->insert(data.data(), data.size());
    }
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      --pending_count_;
    }
    sync_cv_.notify_all();
  }
}

Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }

  std::shared_lock<std::shared_mutex> lock(trie_mutex_);

  std::vector<std::shared_ptr<MatchStateEntry>> state_entries;
  state_entries.reserve(state_ids.size());
  {
    std::lock_guard<std::mutex> state_map_lock(match_state_mutex_);
    for (const auto state_id : state_ids) {
      auto& entry = match_state_[state_id];
      if (!entry) {
        entry = std::make_shared<MatchStateEntry>();
      }
      state_entries.emplace_back(entry);
    }
  }

  using TrieResultBuildFn =
      Result (Trie::*)(const int32_t*, size_t, int32_t, size_t, const Param&, MatchState&, size_t) const;
  using SamResultBuildFn = Result (SuffixAutomaton::*)(const int32_t*, size_t, int32_t, size_t, const Param&) const;
  TrieResultBuildFn trie_result_build_fn;
  SamResultBuildFn sam_result_build_fn;
  if (param_.match_type == "BFS") {
    trie_result_build_fn = &Trie::buildRecency;
    sam_result_build_fn = &SuffixAutomaton::buildRecency;
  } else if (param_.match_type == "PROB") {
    trie_result_build_fn = &Trie::buildFrequency;
    sam_result_build_fn = &SuffixAutomaton::buildFrequency;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  // All budget values are loop-invariant (Trie read lock held, sams_ won't change).
  const size_t num_sams = sams_.size();
  const auto total_draft_token_num = param_.get_draft_token_num(tokens.size());
  const size_t total_sam_budget =
      num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
  const size_t per_sam_budget = num_sams > 0 ? total_sam_budget / num_sams : size_t{0};
  const size_t trie_budget = total_draft_token_num - (per_sam_budget * num_sams);

  Result merged;
  for (size_t i = 0; i < state_ids.size(); ++i) {
    const auto& suffix = tokens[i];
    if (suffix.empty()) {
      throw std::runtime_error("batchMatch received an empty token tail");
    }

    auto& state_entry = state_entries[i];
    std::unique_lock<std::mutex> state_lock(state_entry->mutex);
    auto& state = state_entry->state;

    if (total_sam_budget == 0 || per_sam_budget == 0) {
      auto res = (trie_.get()->*trie_result_build_fn)(
          suffix.data(), suffix.size(), suffix.back(), total_draft_token_num, param_, state, total_lens[i]);
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    auto combined = (trie_.get()->*trie_result_build_fn)(
        suffix.data(), suffix.size(), suffix.back(), trie_budget, param_, state, total_lens[i]);

    for (const auto& [_, sam] : sams_) {
      auto sam_res =
          (sam.get()->*sam_result_build_fn)(suffix.data(), suffix.size(), suffix.back(), per_sam_budget, param_);
      combined = combineRootResults_(suffix.back(), static_cast<int>(total_draft_token_num + 1), combined, sam_res);
    }

    merged.token.insert(merged.token.end(), combined.token.begin(), combined.token.end());
    merged.mask.insert(merged.mask.end(), combined.mask.begin(), combined.mask.end());
  }
  return merged;
}

void Ngram::eraseMatchState(const std::vector<int64_t>& state_ids) {
  std::lock_guard<std::mutex> lock(match_state_mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
  }
}

}  // namespace ngram
