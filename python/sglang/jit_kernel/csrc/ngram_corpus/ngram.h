#pragma once

#include "param.h"
#include "queue.h"
#include "result.h"
#include "suffix_automaton.h"
#include "trie.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ngram {

struct MatchStateEntry {
  std::mutex mutex;
  MatchState state;
};

class Ngram {
  std::unique_ptr<Trie> trie_;
  std::unordered_map<std::string, std::unique_ptr<SuffixAutomaton>> sams_;
  // FIXME: single staging slot — only one corpus can be loaded at a time.
  // To support concurrent loads, move staging into a per-load local variable.
  std::unique_ptr<SuffixAutomaton> staging_sam_;
  Param param_;

  // Readers of trie_ / sams_ can run concurrently. Inserts, resets, and
  // external-corpus publication take the exclusive side of this lock.
  mutable std::shared_mutex trie_mutex_;

  // pending_count_ has a separate lock so synchronize() never waits while
  // holding the Trie write lock.
  mutable std::mutex pending_mutex_;
  mutable std::condition_variable sync_cv_;
  // Counts inserts from enqueue through trie_->insert() completion, not just
  // queue occupancy. A dequeued item may still be mid-insert.
  size_t pending_count_ = 0;
  utils::Queue<std::vector<int32_t>> insert_queue_;
  std::thread insert_worker_;

  // The map lock is held only while resolving state ids. Each request state
  // has its own lock, so unrelated batchMatch calls remain concurrent.
  mutable std::mutex match_state_mutex_;
  std::unordered_map<int64_t, std::shared_ptr<MatchStateEntry>> match_state_;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize() const;

  void asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

  void startExternalCorpusLoad();

  void appendExternalCorpusTokens(const std::vector<int32_t>& tokens);

  // Publishes the staged corpus. Duplicate corpus_id is rejected.
  void finishExternalCorpusLoad(const std::string& corpus_id);

  void removeExternalCorpus(const std::string& corpus_id);

  void resetStagingSam();

  void clearExternalCorpus();

  std::vector<std::pair<std::string, int64_t>> listExternalCorpora() const;

  Result batchMatch(
      const std::vector<int64_t>& state_ids,
      const std::vector<std::vector<int32_t>>& tokens,
      const std::vector<size_t>& total_lens);

  void eraseMatchState(const std::vector<int64_t>& state_ids);

  // Resets the online trie and match state but preserves external corpora
  // (sams_). External corpora are user-managed via add/remove APIs and
  // should not be affected by cache flushes.
  void reset() {
    {
      std::unique_lock<std::shared_mutex> lock(trie_mutex_);
      if (trie_) {
        trie_->reset();
      }
    }
    {
      std::lock_guard<std::mutex> lock(match_state_mutex_);
      match_state_.clear();
    }
  }

  const Param& param() const {
    return param_;
  }

 private:
  void insertWorker();
};

}  // namespace ngram
