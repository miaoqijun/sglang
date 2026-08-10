#pragma once

#include "param.h"
#include "queue.h"
#include "result.h"
#include "suffix_automaton.h"
#include "trie.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ngram {

class Ngram {
  struct InsertTask {
    std::vector<int32_t> tokens;
  };

  struct InsertEpoch {
    uint64_t ticket = 0;
    bool remote = false;
    std::vector<InsertTask> tasks;
  };

  struct EpochCursor {
    std::shared_ptr<const InsertEpoch> epoch;
    size_t next_task = 0;
  };

  std::unique_ptr<Trie> trie_;
  std::unordered_map<std::string, std::unique_ptr<SuffixAutomaton>> sams_;
  // FIXME: single staging slot — only one corpus can be loaded at a time.
  // To support concurrent loads, move staging into a per-load local variable.
  std::unique_ptr<SuffixAutomaton> staging_sam_;
  Param param_;

  // NOTE: protects trie_, sams_, and insertion stats.
  // staging_sam_ is NOT
  // protected by mutex_ — it is only accessed from the corpus loading thread.
  // finishExternalCorpusLoad briefly acquires mutex_ to move the completed
  // SAM into sams_.
  mutable std::mutex mutex_;
  // Work queues are independent from the Trie lock.  Remote work is staged as
  // immutable epochs and released with a constant-size operation.  Local work
  // is always selected first, so waitLocal() cannot be trapped behind a remote
  // backlog. A remote cursor yields after every task, allowing a newly
  // submitted local epoch to preempt it without reordering remote windows.
  mutable std::mutex work_mutex_;
  mutable std::condition_variable work_cv_;
  std::deque<EpochCursor> ready_local_;
  std::list<EpochCursor> staged_remote_;
  std::list<EpochCursor> ready_remote_;
  bool insert_worker_closed_ = false;
  uint64_t next_local_ticket_ = 0;
  uint64_t next_remote_ticket_ = 0;
  uint64_t released_remote_ticket_ = 0;

  mutable std::mutex completion_mutex_;
  mutable std::condition_variable completion_cv_;
  uint64_t completed_local_ticket_ = 0;
  uint64_t completed_remote_ticket_ = 0;
  uint64_t local_epoch_tasks_ = 0;
  uint64_t remote_epoch_tasks_ = 0;
  uint64_t local_epoch_edge_visits_ = 0;
  uint64_t remote_epoch_edge_visits_ = 0;
  uint64_t local_epoch_service_ns_ = 0;
  uint64_t remote_epoch_service_ns_ = 0;
  std::thread insert_worker_;
  std::unordered_map<int64_t, MatchState> match_state_;
  uint64_t window_insert_tasks_ = 0;
  uint64_t window_insert_ns_ = 0;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize();

  void waitLocal(uint64_t ticket) const;

  void waitRemote(uint64_t ticket) const;

  bool remoteReady(uint64_t ticket) const;

  uint64_t asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

  // Called from the history stager thread after the window CSR has been copied
  // into native storage. The epoch remains frozen until releaseRemoteEpochs().
  uint64_t stageRemoteEpoch(std::vector<std::vector<int32_t>>&& windows);

  // Atomically freezes the current staged prefix and makes it executable.
  // list::splice moves all epoch cursors without walking their payloads.
  uint64_t releaseRemoteEpochs();

  std::string insertStatsJson() const;

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
    std::unique_lock<std::mutex> lock(mutex_);
    if (trie_) {
      trie_->reset();
    }
    match_state_.clear();
    window_insert_tasks_ = 0;
    window_insert_ns_ = 0;
  }

  const Param& param() const {
    return param_;
  }

 private:
  uint64_t submitLocalEpoch(std::vector<InsertTask>&& tasks);
  void insertWorker();
};

}  // namespace ngram
