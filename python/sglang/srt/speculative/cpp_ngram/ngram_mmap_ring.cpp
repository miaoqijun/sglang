#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;

namespace {

constexpr uint64_t kMagic = 0x53474C4E4752494EULL;  // "SGLNGRIN"
// Internal ABI marker only. There is one window-record layout and no
// application-level record versioning.
constexpr uint32_t kAbi = 1;
constexpr size_t kSuperblockSize = 4096;
constexpr uint32_t kSlotSize = 128;
constexpr uint32_t kMaxTokens = 18;
constexpr uint64_t kBusy = std::numeric_limits<uint64_t>::max();

struct alignas(64) RingHeader {
  uint64_t magic;
  uint32_t abi;
  uint32_t slot_size;
  uint64_t capacity;
  uint64_t namespace_hash;
  uint64_t max_tokens;
  uint8_t padding0[24];

  alignas(64) uint64_t epoch;
  uint8_t padding1[56];

  alignas(64) uint64_t write_seq;
  uint8_t padding2[56];

  alignas(64) uint64_t ready;
  uint8_t padding3[56];
};

static_assert(sizeof(RingHeader) == 256);

struct alignas(128) WindowSlot {
  uint64_t commit_seq;
  uint8_t token_count;
  uint8_t reserved[23];
  int32_t tokens[kMaxTokens];
  uint8_t padding[24];
};

static_assert(sizeof(WindowSlot) == kSlotSize);
static_assert(alignof(WindowSlot) == 128);
static_assert(offsetof(WindowSlot, tokens) == 32);
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

inline uint64_t atomic_load_acquire(const uint64_t* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline void atomic_store_release(uint64_t* ptr, uint64_t value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

inline void atomic_store_seq_cst(uint64_t* ptr, uint64_t value) {
  __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
}

inline bool valid_window_snapshot(const WindowSlot& snapshot) {
  return snapshot.token_count > 0 && snapshot.token_count <= kMaxTokens;
}

inline size_t mapping_size(uint64_t capacity) {
  if (capacity == 0 ||
      capacity > (std::numeric_limits<size_t>::max() - kSuperblockSize) /
                     kSlotSize) {
    throw std::invalid_argument("invalid mmap ring capacity");
  }
  return kSuperblockSize + static_cast<size_t>(capacity) * kSlotSize;
}

inline void throw_errno(const std::string& operation,
                        const std::string& path) {
  throw std::runtime_error(operation + " failed for " + path + ": " +
                           std::strerror(errno));
}

class RingWriter {
 public:
  RingWriter(const std::string& path, uint64_t capacity,
             uint64_t namespace_hash, uint64_t epoch)
      : path_(path), capacity_(capacity), epoch_(epoch) {
    if (epoch == 0 || epoch == kBusy) {
      throw std::invalid_argument("writer epoch must be a nonzero finite value");
    }
    size_ = mapping_size(capacity_);
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd_ < 0) throw_errno("open", path_);
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      int saved = errno;
      ::close(fd_);
      fd_ = -1;
      errno = saved;
      throw_errno("exclusive writer lock", path_);
    }
    if (::ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
      close();
      throw_errno("ftruncate", path_);
    }
    mapping_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      close();
      throw_errno("mmap", path_);
    }

    auto* old_header = static_cast<RingHeader*>(mapping_);
    atomic_store_seq_cst(&old_header->ready, 0);
    std::memset(mapping_, 0, size_);

    header_ = static_cast<RingHeader*>(mapping_);
    slots_ = reinterpret_cast<WindowSlot*>(
        static_cast<uint8_t*>(mapping_) + kSuperblockSize);
    header_->magic = kMagic;
    header_->abi = kAbi;
    header_->slot_size = kSlotSize;
    header_->capacity = capacity_;
    header_->namespace_hash = namespace_hash;
    header_->max_tokens = kMaxTokens;
    header_->epoch = epoch_;
    atomic_store_release(&header_->write_seq, 0);
    atomic_store_release(&header_->ready, 1);
  }

  RingWriter(const RingWriter&) = delete;
  RingWriter& operator=(const RingWriter&) = delete;

  ~RingWriter() { close_locked(); }

  size_t publish_windows_csr(
      py::array_t<int32_t, py::array::c_style> flat,
      py::array_t<int64_t, py::array::c_style> offsets) {
    // Request and retain both buffers while holding the GIL.  The buffer_info
    // objects own the exported buffers until this method returns; all metadata
    // and payload validation below is native and runs without the GIL.
    py::buffer_info flat_info = flat.request(false);
    py::buffer_info offsets_info = offsets.request(false);
    const int32_t* flat_ptr = static_cast<const int32_t*>(flat_info.ptr);
    const int64_t* offsets_ptr =
        static_cast<const int64_t*>(offsets_info.ptr);

    size_t published_count = 0;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
      published_count = publish_windows_csr_unlocked(
          flat_info, offsets_info, flat_ptr, offsets_ptr);
    }
    return published_count;
  }

  uint64_t epoch() const { return epoch_; }
  uint64_t capacity() const { return capacity_; }

  void close() {
    // A GIL-free publish may currently own lifecycle_mutex_.  Do not wait for
    // it while holding the GIL: publish must reacquire the GIL before returning.
    py::gil_scoped_release release;
    close_locked();
  }

 private:
  static void ensure_sequence_space(uint64_t seq, size_t count) {
    // kBusy is reserved as an in-progress slot marker and must never become a
    // committed sequence number.  In practice this guard is unreachable, but
    // it prevents wraparound from corrupting the ring invariants.
    if (seq >= kBusy - 1 ||
        count > static_cast<size_t>((kBusy - 1) - seq)) {
      throw std::overflow_error("mmap ring sequence space exhausted");
    }
  }

  void ensure_ready_unlocked() const {
    if (header_ == nullptr || slots_ == nullptr ||
        atomic_load_acquire(&header_->ready) != 1) {
      throw std::runtime_error("ring writer is closed or not ready");
    }
  }

  size_t publish_windows_csr_unlocked(
      const py::buffer_info& flat_info,
      const py::buffer_info& offsets_info, const int32_t* flat,
      const int64_t* offsets) {
    if (flat_info.ndim != 1 || offsets_info.ndim != 1) {
      throw std::invalid_argument(
          "publish_windows_csr buffers must be one-dimensional");
    }
    if (flat_info.size < 0 || offsets_info.size < 1 ||
        (flat_info.size > 0 && flat == nullptr) || offsets == nullptr) {
      throw std::invalid_argument(
          "publish_windows_csr requires readable buffers and offsets");
    }

    const size_t flat_size = static_cast<size_t>(flat_info.size);
    const size_t offsets_size = static_cast<size_t>(offsets_info.size);
    const size_t window_count = offsets_size - 1;
    if (flat_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument(
          "publish_windows_csr flat buffer is too large");
    }
    if (offsets[0] != 0 ||
        offsets[window_count] != static_cast<int64_t>(flat_size)) {
      throw std::invalid_argument(
          "publish_windows_csr offsets must span the flat buffer");
    }

    // Validate the complete CSR before publishing anything.  This makes bad
    // input fail without exposing a prefix of the batch to readers.
    for (size_t i = 0; i < window_count; ++i) {
      const int64_t begin = offsets[i];
      const int64_t end = offsets[i + 1];
      if (begin < 0 || end <= begin ||
          end > static_cast<int64_t>(flat_size) ||
          end - begin > static_cast<int64_t>(kMaxTokens)) {
        throw std::invalid_argument(
            "publish_windows_csr window lengths must be in [1, 18]");
      }
    }
    if (window_count == 0) return 0;

    ensure_ready_unlocked();
    uint64_t seq = atomic_load_acquire(&header_->write_seq);
    ensure_sequence_space(seq, window_count);
    for (size_t i = 0; i < window_count; ++i) {
      const size_t begin = static_cast<size_t>(offsets[i]);
      const size_t count =
          static_cast<size_t>(offsets[i + 1] - offsets[i]);
      ++seq;
      WindowSlot* slot = &slots_[(seq - 1) % capacity_];
      atomic_store_seq_cst(&slot->commit_seq, kBusy);
      slot->token_count = static_cast<uint8_t>(count);
      std::memcpy(slot->tokens, flat + begin, count * sizeof(int32_t));
      atomic_store_release(&slot->commit_seq, seq);
    }
    atomic_store_release(&header_->write_seq, seq);
    return window_count;
  }

  void close_locked() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (header_ != nullptr) {
      atomic_store_seq_cst(&header_->ready, 0);
    }
    if (mapping_ != nullptr) {
      ::munmap(mapping_, size_);
      mapping_ = nullptr;
      header_ = nullptr;
      slots_ = nullptr;
    }
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
    }
  }

  std::string path_;
  int fd_ = -1;
  void* mapping_ = nullptr;
  size_t size_ = 0;
  uint64_t capacity_ = 0;
  uint64_t epoch_ = 0;
  RingHeader* header_ = nullptr;
  WindowSlot* slots_ = nullptr;
  mutable std::mutex lifecycle_mutex_;
};

class RingReader {
 public:
  RingReader(const std::string& path, uint64_t expected_namespace_hash,
             bool start_from_beginning)
      : path_(path),
        expected_namespace_hash_(expected_namespace_hash),
        start_from_beginning_(start_from_beginning) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw_errno("open", path_);

    // The writer holds an exclusive lifetime lock.  Acquiring a shared lock
    // means this is a stale file with no live producer.
    if (::flock(fd_, LOCK_SH | LOCK_NB) == 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap ring has no live writer: " + path_);
    }
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
      int saved = errno;
      ::close(fd_);
      fd_ = -1;
      errno = saved;
      throw_errno("reader writer-lock probe", path_);
    }

    struct stat stat_buf {};
    if (::fstat(fd_, &stat_buf) != 0) {
      close();
      throw_errno("fstat", path_);
    }
    if (stat_buf.st_size < static_cast<off_t>(kSuperblockSize + kSlotSize)) {
      close();
      throw std::runtime_error("mmap ring file is too small: " + path_);
    }
    size_ = static_cast<size_t>(stat_buf.st_size);
    mapping_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      close();
      throw_errno("mmap", path_);
    }
    header_ = static_cast<const RingHeader*>(mapping_);
    if (atomic_load_acquire(&header_->ready) != 1) {
      close();
      throw std::runtime_error("mmap ring is not ready: " + path_);
    }
    try {
      validate_header();
    } catch (...) {
      close();
      throw;
    }
    slots_ = reinterpret_cast<const WindowSlot*>(
        static_cast<const uint8_t*>(mapping_) + kSuperblockSize);
    epoch_ = atomic_load_acquire(&header_->epoch);
    uint64_t published = atomic_load_acquire(&header_->write_seq);
    cursor_ = start_from_beginning_
                  ? (published > capacity_ ? published - capacity_ : 0)
                  : published;
  }

  RingReader(const RingReader&) = delete;
  RingReader& operator=(const RingReader&) = delete;

  ~RingReader() { close_locked(); }

  py::tuple poll_into(
      py::array_t<int32_t, py::array::c_style> flat,
      py::array_t<int64_t, py::array::c_style> offsets,
      size_t record_offset, size_t token_offset, size_t record_limit,
      size_t token_limit) {
    if (flat.ndim() != 1 || offsets.ndim() != 1) {
      throw std::invalid_argument("poll_into buffers must be one-dimensional");
    }
    if (!flat.writeable() || !offsets.writeable()) {
      throw std::invalid_argument("poll_into buffers must be writable");
    }
    if (record_offset > record_limit || token_offset > token_limit) {
      throw std::invalid_argument("poll_into offsets exceed their limits");
    }
    const size_t flat_size = static_cast<size_t>(flat.size());
    const size_t offsets_size = static_cast<size_t>(offsets.size());
    // offsets contains CSR boundaries, so an exclusive record_limit requires
    // offsets[record_limit] to exist even when no records are produced.
    if (token_limit > flat_size || record_limit >= offsets_size) {
      throw std::invalid_argument("poll_into limits exceed buffer capacities");
    }

    int32_t* flat_ptr = flat.mutable_data();
    int64_t* offsets_ptr = offsets.mutable_data();
    PackedPollResult result;
    {
      // Keep the mapping alive for the complete native scan.  The lifecycle
      // lock is destroyed before gil_scoped_release tries to reacquire the
      // GIL, so a concurrent close cannot deadlock while waiting for it.
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
      result = poll_into_unlocked(flat_ptr, offsets_ptr, record_offset,
                                  token_offset, record_limit, token_limit);
    }
    return py::make_tuple(result.produced, result.token_count, result.epoch,
                          result.reset, result.gaps, result.cursor,
                          result.published);
  }

  bool writer_alive() const {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (fd_ < 0) return false;
    if (::flock(fd_, LOCK_SH | LOCK_NB) == 0) {
      ::flock(fd_, LOCK_UN);
      return false;
    }
    return errno == EWOULDBLOCK || errno == EAGAIN;
  }

  uint64_t epoch() const {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    return epoch_;
  }
  uint64_t capacity() const {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    return capacity_;
  }
  uint64_t cursor() const {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    return cursor_;
  }

  bool possibly_ready_for_wait() const {
    // Waiting must not become an unbounded stop/close barrier behind a long
    // poll.  Contention is therefore treated as a possible state change and
    // produces a harmless spurious wakeup instead of blocking on the reader.
    std::unique_lock<std::mutex> lifecycle_lock(
        lifecycle_mutex_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) return true;

    // A locally closed reader, a writer reset/close, an epoch replacement, or
    // any write_seq different from the reader cursor all require the Python
    // pull loop to run its normal poll/health handling.
    if (header_ == nullptr) return true;
    const uint64_t ready_before = atomic_load_acquire(&header_->ready);
    if (ready_before != 1) return true;
    const uint64_t current_epoch = atomic_load_acquire(&header_->epoch);
    const uint64_t published = atomic_load_acquire(&header_->write_seq);
    const uint64_t ready_after = atomic_load_acquire(&header_->ready);
    return ready_after != 1 || current_epoch != epoch_ || published != cursor_;
  }

  void close() {
    // A GIL-free poll may currently own lifecycle_mutex_.  Do not wait for it
    // while holding the GIL: poll must reacquire the GIL before returning.
    py::gil_scoped_release release;
    close_locked();
  }

 private:
  struct PackedPollResult {
    size_t produced = 0;
    size_t token_count = 0;
    uint64_t epoch = 0;
    bool reset = false;
    size_t gaps = 0;
    uint64_t cursor = 0;
    uint64_t published = 0;
  };

  PackedPollResult poll_into_unlocked(
      int32_t* flat, int64_t* offsets, size_t record_offset,
      size_t token_offset, size_t record_limit, size_t token_limit) {
    PackedPollResult result;
    result.epoch = epoch_;
    result.cursor = cursor_;
    result.published = cursor_;
    offsets[record_offset] = static_cast<int64_t>(token_offset);

    if (record_offset == record_limit || header_ == nullptr ||
        atomic_load_acquire(&header_->ready) != 1) {
      return result;
    }

    uint64_t current_epoch = atomic_load_acquire(&header_->epoch);
    uint64_t published = atomic_load_acquire(&header_->write_seq);
    if (current_epoch != epoch_) {
      epoch_ = current_epoch;
      cursor_ = start_from_beginning_
                    ? (published > capacity_ ? published - capacity_ : 0)
                    : published;
      result.reset = true;
    }
    if (published < cursor_) {
      cursor_ = start_from_beginning_
                    ? (published > capacity_ ? published - capacity_ : 0)
                    : published;
      result.reset = true;
    }
    if (published - cursor_ > capacity_) {
      cursor_ = published - capacity_;
      result.reset = true;
      ++result.gaps;
    }

    size_t write_token = token_offset;
    const size_t record_capacity = record_limit - record_offset;
    while (cursor_ < published && result.produced < record_capacity) {
      uint64_t seq = cursor_ + 1;
      const WindowSlot* slot = &slots_[(seq - 1) % capacity_];
      uint64_t before = atomic_load_acquire(&slot->commit_seq);
      if (before == kBusy || before < seq) {
        break;
      }
      if (before != seq) {
        cursor_ = seq;
        result.reset = true;
        ++result.gaps;
        continue;
      }

      WindowSlot snapshot;
      std::memcpy(&snapshot, slot, sizeof(snapshot));
      uint64_t after = atomic_load_acquire(&slot->commit_seq);
      const bool valid = after == before && snapshot.commit_seq == before &&
                         valid_window_snapshot(snapshot);
      if (!valid) {
        cursor_ = seq;
        result.reset = true;
        ++result.gaps;
        continue;
      }

      const size_t count = snapshot.token_count;
      if (count > token_limit - write_token) {
        break;
      }
      std::memcpy(flat + write_token, snapshot.tokens,
                  count * sizeof(int32_t));
      write_token += count;
      ++result.produced;
      offsets[record_offset + result.produced] =
          static_cast<int64_t>(write_token);
      cursor_ = seq;
    }

    result.token_count = write_token - token_offset;
    result.epoch = epoch_;
    result.cursor = cursor_;
    result.published = published;
    return result;
  }

  void close_locked() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (mapping_ != nullptr) {
      ::munmap(mapping_, size_);
      mapping_ = nullptr;
      header_ = nullptr;
      slots_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  void validate_header() {
    if (header_->magic != kMagic || header_->abi != kAbi ||
        header_->slot_size != kSlotSize ||
        header_->namespace_hash != expected_namespace_hash_ ||
        header_->max_tokens != kMaxTokens) {
      throw std::runtime_error("incompatible mmap ring header: " + path_);
    }
    capacity_ = header_->capacity;
    if (mapping_size(capacity_) != size_) {
      throw std::runtime_error("mmap ring size/header mismatch: " + path_);
    }
  }

  std::string path_;
  uint64_t expected_namespace_hash_ = 0;
  bool start_from_beginning_ = false;
  int fd_ = -1;
  void* mapping_ = nullptr;
  size_t size_ = 0;
  uint64_t capacity_ = 0;
  uint64_t epoch_ = 0;
  uint64_t cursor_ = 0;
  const RingHeader* header_ = nullptr;
  const WindowSlot* slots_ = nullptr;
  mutable std::mutex lifecycle_mutex_;
};

py::tuple wait_readers(const py::list& raw_readers, int64_t timeout_ms,
                       int64_t scan_interval_ms) {
  if (timeout_ms < 0) {
    throw std::invalid_argument("wait_readers timeout_ms must be nonnegative");
  }
  if (scan_interval_ms <= 0) {
    throw std::invalid_argument(
        "wait_readers scan_interval_ms must be positive");
  }

  // Keep an owning Python reference to every reader while native code uses
  // raw pointers without the GIL.  This also remains safe if another thread
  // mutates the caller's list after this function releases the GIL.
  std::vector<py::object> reader_owners;
  std::vector<RingReader*> readers;
  reader_owners.reserve(raw_readers.size());
  readers.reserve(raw_readers.size());
  size_t index = 0;
  for (py::handle item : raw_readers) {
    RingReader* reader = nullptr;
    try {
      reader = py::cast<RingReader*>(item);
    } catch (const py::cast_error&) {
      throw py::type_error("wait_readers item " + std::to_string(index) +
                           " is not a RingReader");
    }
    if (reader == nullptr) {
      throw py::type_error("wait_readers item " + std::to_string(index) +
                           " is not a live RingReader");
    }
    reader_owners.emplace_back(py::reinterpret_borrow<py::object>(item));
    readers.push_back(reader);
    ++index;
  }

  bool ready = false;
  size_t scans = 0;
  {
    py::gil_scoped_release release;
    using Clock = std::chrono::steady_clock;
    const auto timeout = std::chrono::milliseconds(timeout_ms);
    const auto scan_interval = std::chrono::duration_cast<Clock::duration>(
        std::chrono::milliseconds(scan_interval_ms));
    const auto deadline = Clock::now() + timeout;

    // Always perform one immediate scan, including for timeout_ms == 0.
    while (true) {
      ++scans;
      for (RingReader* reader : readers) {
        if (reader->possibly_ready_for_wait()) {
          ready = true;
          break;
        }
      }
      if (ready) break;

      const auto now = Clock::now();
      if (now >= deadline) break;
      std::this_thread::sleep_for(std::min(scan_interval, deadline - now));
    }
  }
  return py::make_tuple(ready, scans);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.attr("SLOT_SIZE") = kSlotSize;
  module.attr("MAX_TOKENS") = kMaxTokens;
  module.attr("SUPERBLOCK_SIZE") = kSuperblockSize;
  py::class_<RingWriter>(module, "RingWriter")
      .def(py::init<const std::string&, uint64_t, uint64_t, uint64_t>())
      .def("publish_windows_csr", &RingWriter::publish_windows_csr,
           py::arg("flat").noconvert(), py::arg("offsets").noconvert())
      .def("close", &RingWriter::close)
      .def_property_readonly("epoch", &RingWriter::epoch)
      .def_property_readonly("capacity", &RingWriter::capacity);

  py::class_<RingReader>(module, "RingReader")
      .def(py::init<const std::string&, uint64_t, bool>())
      .def("poll_into", &RingReader::poll_into, py::arg("flat"),
           py::arg("offsets"), py::arg("record_offset"),
           py::arg("token_offset"), py::arg("record_limit"),
           py::arg("token_limit"))
      .def("writer_alive", &RingReader::writer_alive)
      .def("close", &RingReader::close)
      .def_property_readonly("epoch", &RingReader::epoch)
      .def_property_readonly("capacity", &RingReader::capacity)
      .def_property_readonly("cursor", &RingReader::cursor);

  module.def("wait_readers", &wait_readers, py::arg("readers"),
             py::arg("timeout_ms") = 200,
             py::arg("scan_interval_ms") = 5,
             "Wait without the GIL until any RingReader may need polling.");
}
