#include "ngram.h"

#include "trie.h"
#include <chrono>
#include <limits>
#include <sstream>
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
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    insert_worker_closed_ = true;
  }
  work_cv_.notify_all();
  if (insert_worker_.joinable()) {
    insert_worker_.join();
  }
}

void Ngram::synchronize() {
  uint64_t local_ticket;
  uint64_t remote_ticket;
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (!staged_remote_.empty()) {
      released_remote_ticket_ = staged_remote_.back().epoch->ticket;
      ready_remote_.splice(ready_remote_.end(), staged_remote_);
    }
    local_ticket = next_local_ticket_;
    remote_ticket = released_remote_ticket_;
  }
  work_cv_.notify_one();
  waitLocal(local_ticket);
  waitRemote(remote_ticket);
}

void Ngram::waitLocal(uint64_t ticket) const {
  if (ticket == 0) return;
  std::unique_lock<std::mutex> lock(completion_mutex_);
  completion_cv_.wait(lock, [this, ticket] {
    return completed_local_ticket_ >= ticket;
  });
}

void Ngram::waitRemote(uint64_t ticket) const {
  if (ticket == 0) return;
  std::unique_lock<std::mutex> lock(completion_mutex_);
  completion_cv_.wait(lock, [this, ticket] {
    return completed_remote_ticket_ >= ticket;
  });
}

bool Ngram::remoteReady(uint64_t ticket) const {
  if (ticket == 0) return true;
  std::lock_guard<std::mutex> lock(completion_mutex_);
  return completed_remote_ticket_ >= ticket;
}

uint64_t Ngram::submitLocalEpoch(std::vector<InsertTask>&& tasks) {
  if (tasks.empty()) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    return next_local_ticket_;
  }
  uint64_t ticket;
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (insert_worker_closed_) {
      throw std::runtime_error("NGRAM insert worker is closed");
    }
    ticket = ++next_local_ticket_;
    auto epoch = std::make_shared<InsertEpoch>();
    epoch->ticket = ticket;
    epoch->remote = false;
    epoch->tasks = std::move(tasks);
    ready_local_.push_back(EpochCursor{std::move(epoch), 0});
  }
  work_cv_.notify_one();
  return ticket;
}

uint64_t Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  std::vector<InsertTask> tasks;
  tasks.reserve(tokens.size());
  for (auto&& token : tokens) {
    tasks.push_back(InsertTask{std::move(token)});
  }
  return submitLocalEpoch(std::move(tasks));
}

uint64_t Ngram::stageRemoteEpoch(std::vector<std::vector<int32_t>>&& windows) {
  std::vector<InsertTask> tasks;
  tasks.reserve(windows.size());
  for (auto&& tokens : windows) {
    tasks.push_back(InsertTask{std::move(tokens)});
  }
  if (tasks.empty()) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    return next_remote_ticket_;
  }

  uint64_t ticket;
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (insert_worker_closed_) {
      throw std::runtime_error("NGRAM insert worker is closed");
    }
    ticket = ++next_remote_ticket_;
    auto epoch = std::make_shared<InsertEpoch>();
    epoch->ticket = ticket;
    epoch->remote = true;
    epoch->tasks = std::move(tasks);
    staged_remote_.push_back(EpochCursor{std::move(epoch), 0});
  }
  return ticket;
}

uint64_t Ngram::releaseRemoteEpochs() {
  uint64_t ticket = 0;
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (!staged_remote_.empty()) {
      released_remote_ticket_ = staged_remote_.back().epoch->ticket;
      ready_remote_.splice(ready_remote_.end(), staged_remote_);
      ticket = released_remote_ticket_;
    }
  }
  work_cv_.notify_one();
  return ticket;
}

std::string Ngram::insertStatsJson() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::unique_lock<std::mutex> work_lock(work_mutex_);
  std::unique_lock<std::mutex> completion_lock(completion_mutex_);
  const auto& stats = trie_->insertStats();
  std::ostringstream out;
  out << "{"
      << "\"window_tasks\":" << window_insert_tasks_ << ","
      << "\"window_insert_ns\":" << window_insert_ns_ << ","
      << "\"local_submitted_ticket\":" << next_local_ticket_ << ","
      << "\"local_completed_ticket\":" << completed_local_ticket_ << ","
      << "\"remote_staged_ticket\":" << next_remote_ticket_ << ","
      << "\"remote_released_ticket\":" << released_remote_ticket_ << ","
      << "\"remote_completed_ticket\":" << completed_remote_ticket_ << ","
      << "\"staged_remote_epochs\":" << staged_remote_.size() << ","
      << "\"ready_remote_epochs\":" << ready_remote_.size() << ","
      << "\"ready_local_epochs\":" << ready_local_.size() << ","
      << "\"local_epoch_tasks\":" << local_epoch_tasks_ << ","
      << "\"remote_epoch_tasks\":" << remote_epoch_tasks_ << ","
      << "\"local_epoch_edge_visits\":" << local_epoch_edge_visits_ << ","
      << "\"remote_epoch_edge_visits\":" << remote_epoch_edge_visits_ << ","
      << "\"local_epoch_service_ns\":" << local_epoch_service_ns_ << ","
      << "\"remote_epoch_service_ns\":" << remote_epoch_service_ns_ << ","
      << "\"window_records\":" << stats.window_records << ","
      << "\"window_edge_visits\":" << stats.window_edge_visits << ","
      << "\"squeeze_calls\":" << stats.squeeze_calls << ","
      << "\"squeezed_nodes\":" << stats.squeezed_nodes
      << "}";
  return out.str();
}

// NOTE: staging operations (start/append/finish) are called from a background
// thread during async corpus loading. They do NOT hold mutex_ because
// staging_sam_ is disjoint from sams_ / trie_. Only finishExternalCorpusLoad
// briefly acquires mutex_ when moving the completed SAM into sams_.
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
  std::unique_lock<std::mutex> lock(mutex_);
  if (sams_.find(corpus_id) != sams_.end()) {
    throw std::runtime_error(
        "External corpus '" + corpus_id + "' already exists. Remove it before adding a new corpus with the same id.");
  }
  sams_.emplace(corpus_id, std::move(staging_sam_));
}

void Ngram::removeExternalCorpus(const std::string& corpus_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  sams_.erase(corpus_id);
}

void Ngram::resetStagingSam() {
  // staging_sam_ is only accessed from the loading thread — no lock needed.
  staging_sam_.reset();
}

void Ngram::clearExternalCorpus() {
  std::unique_lock<std::mutex> lock(mutex_);
  sams_.clear();
  staging_sam_.reset();
}

std::vector<std::pair<std::string, int64_t>> Ngram::listExternalCorpora() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::pair<std::string, int64_t>> entries;
  entries.reserve(sams_.size());
  for (const auto& [id, sam] : sams_) {
    entries.emplace_back(id, sam->tokenCount());
  }
  return entries;
}

void Ngram::insertWorker() {
  for (;;) {
    std::shared_ptr<const InsertEpoch> epoch;
    size_t task_index = 0;
    bool epoch_complete = false;
    {
      std::unique_lock<std::mutex> lock(work_mutex_);
      work_cv_.wait(lock, [this] {
        return insert_worker_closed_ || !ready_local_.empty() || !ready_remote_.empty();
      });
      if (insert_worker_closed_) break;

      // Re-evaluate this choice after every task.  This is the key invariant
      // that keeps local progress independent of an arbitrarily long remote
      // epoch while preserving FIFO order within the remote domain.
      auto* cursor = !ready_local_.empty() ? &ready_local_.front() : &ready_remote_.front();
      epoch = cursor->epoch;
      task_index = cursor->next_task++;
      epoch_complete = cursor->next_task == epoch->tasks.size();
      if (epoch_complete) {
        if (epoch->remote) {
          ready_remote_.pop_front();
        } else {
          ready_local_.pop_front();
        }
      }
    }

    const auto& task = epoch->tasks[task_index];
    const auto start = std::chrono::steady_clock::now();
    uint64_t edge_visits_delta = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      const uint64_t before_edge_visits = trie_->insertStats().window_edge_visits;
      trie_->insert(task.tokens.data(), task.tokens.size());
      ++window_insert_tasks_;
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start);
      window_insert_ns_ += static_cast<uint64_t>(elapsed.count());
      edge_visits_delta = trie_->insertStats().window_edge_visits - before_edge_visits;
    }
    const auto service_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    {
      std::lock_guard<std::mutex> lock(completion_mutex_);
      if (epoch->remote) {
        ++remote_epoch_tasks_;
        remote_epoch_edge_visits_ += edge_visits_delta;
        remote_epoch_service_ns_ += service_ns;
        if (epoch_complete) completed_remote_ticket_ = epoch->ticket;
      } else {
        ++local_epoch_tasks_;
        local_epoch_edge_visits_ += edge_visits_delta;
        local_epoch_service_ns_ += service_ns;
        if (epoch_complete) completed_local_ticket_ = epoch->ticket;
      }
    }
    if (epoch_complete) completion_cv_.notify_all();
  }
}

Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }

  std::unique_lock<std::mutex> lock(mutex_);

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

  // All budget values are loop-invariant (mutex_ held, sams_ won't change).
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

    auto& state = match_state_[state_ids[i]];

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
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
  }
}

}  // namespace ngram
