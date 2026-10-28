#include <algorithm>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/exception.h"
#include "frontend/frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class TimedTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd, TimedTrace, "TimedTrace",
      "Timed read trace with source identity, release cycles, callbacks, and drain.")

 private:
  struct TraceEntry {
    uint64_t request_id;
    Clk_t release_cycle;
    int source_id;
    Addr_t address;
  };

  struct Completion {
    Clk_t accepted_cycle = -1;
    Clk_t completion_cycle = -1;
  };

  std::vector<TraceEntry> m_trace;
  std::vector<Completion> m_completions;
  std::deque<size_t> m_pending;
  size_t m_next_trace_idx = 0;
  size_t m_submitted = 0;
  size_t m_completed = 0;
  uint m_issue_width = 0;
  uint m_source_issue_width = 1;
  int m_num_sources = 1;
  std::string m_completion_path;

 public:
  void init() override {
    const auto trace_path = param<std::string>("path").desc("Timed trace path.").required();
    m_completion_path =
        param<std::string>("completion_path").desc("Completion CSV path.").required();
    m_clock_ratio = param<uint>("clock_ratio").default_val(1);
    m_issue_width = param<uint>("issue_width").default_val(0);
    m_source_issue_width = param<uint>("source_issue_width").default_val(1);
    load_trace(trace_path);
    m_completions.resize(m_trace.size());
  }

  void tick() override {
    while (m_next_trace_idx < m_trace.size() &&
           m_trace[m_next_trace_idx].release_cycle <= m_clk) {
      m_pending.push_back(m_next_trace_idx++);
    }

    std::unordered_map<int, uint> issued_by_source;
    size_t attempts = m_pending.size();
    if (m_issue_width > 0) {
      attempts = std::min(attempts, static_cast<size_t>(m_issue_width));
    }
    uint issued = 0;
    for (size_t attempt = 0; attempt < attempts; ++attempt) {
      const auto index = m_pending.front();
      m_pending.pop_front();
      const auto& entry = m_trace[index];
      if ((m_source_issue_width > 0 &&
           issued_by_source[entry.source_id] >= m_source_issue_width)) {
        m_pending.push_back(index);
        continue;
      }

      Request request(
          entry.address, Request::Type::Read, entry.source_id,
          [this, index](Request& completed) { receive(index, completed); });
      if (m_memory_system->send(request)) {
        m_completions[index].accepted_cycle = m_clk;
        ++m_submitted;
        ++issued;
        ++issued_by_source[entry.source_id];
      } else {
        m_pending.push_back(index);
      }
    }
    ++m_clk;
  }

  bool is_finished() override {
    return m_next_trace_idx == m_trace.size() && m_pending.empty() &&
           m_submitted == m_trace.size() && m_completed == m_trace.size();
  }

  int get_num_cores() override { return m_num_sources; }

  void finalize() override {
    if (!is_finished()) {
      throw std::runtime_error("TimedTrace finalized before every request completed");
    }
    fs::path output(m_completion_path);
    if (output.has_parent_path()) {
      fs::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream.is_open()) {
      throw ConfigurationError("Cannot open TimedTrace completion path {}", m_completion_path);
    }
    stream << "request_id,source_id,release_cycle,accepted_cycle,completion_cycle,address\n";
    for (size_t index = 0; index < m_trace.size(); ++index) {
      const auto& entry = m_trace[index];
      const auto& completion = m_completions[index];
      stream << entry.request_id << ',' << entry.source_id << ','
             << entry.release_cycle << ',' << completion.accepted_cycle << ','
             << completion.completion_cycle << ",0x" << std::hex
             << entry.address << std::dec << '\n';
    }
    IFrontEnd::finalize();
  }

 private:
  void load_trace(const std::string& trace_path) {
    std::ifstream stream(trace_path);
    if (!stream.is_open()) {
      throw ConfigurationError("Timed trace {} cannot be opened", trace_path);
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
      ++line_number;
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream tokens(line);
      std::string address_token;
      TraceEntry entry{};
      if (!(tokens >> entry.request_id >> entry.release_cycle >> entry.source_id >>
            address_token)) {
        throw ConfigurationError("Timed trace {} line {} is invalid", trace_path,
                                 line_number);
      }
      entry.address = std::stoll(address_token, nullptr, 0);
      if (entry.release_cycle < 0 || entry.source_id < 0 || entry.address < 0) {
        throw ConfigurationError("Timed trace {} line {} has a negative field",
                                 trace_path, line_number);
      }
      m_num_sources = std::max(m_num_sources, entry.source_id + 1);
      m_trace.push_back(entry);
    }
    if (!std::is_sorted(
            m_trace.begin(), m_trace.end(),
            [](const TraceEntry& left, const TraceEntry& right) {
              return left.release_cycle < right.release_cycle ||
                     (left.release_cycle == right.release_cycle &&
                      left.request_id < right.request_id);
            })) {
      throw ConfigurationError(
          "Timed trace {} must be sorted by release_cycle then request_id", trace_path);
    }
  }

  void receive(const size_t index, Request& request) {
    if (m_completions[index].completion_cycle >= 0) {
      throw std::runtime_error("TimedTrace request completed more than once");
    }
    m_completions[index].completion_cycle = request.depart;
    ++m_completed;
  }
};

}  // namespace Ramulator
