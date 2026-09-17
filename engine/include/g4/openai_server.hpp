#pragma once

#include "g4/service.hpp"

#include <cstdint>

namespace g4 {

// OpenAI Chat Completions transport around one GPU-owning scheduler. Network
// threads never touch CUDA; they enqueue work and wait for the scheduler.
class OpenAiServer {
 public:
  explicit OpenAiServer(MultimodalRuntime& runtime, int maximum_batch = 8);
  ~OpenAiServer();
  OpenAiServer(const OpenAiServer&) = delete;
  OpenAiServer& operator=(const OpenAiServer&) = delete;

  [[noreturn]] void run(std::uint16_t port = 8080);

 private:
  struct Impl;
  Impl* impl_{};
};

}  // namespace g4
