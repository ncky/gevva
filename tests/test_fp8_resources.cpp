#include "gpu/fp8_resources.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>

struct Runtime {
  using Event = void*;
  static inline int fail_at = 0, calls = 0;
  static inline std::unordered_set<void*> buffers, events;
  static void step() {
    if (++calls == fail_at) throw std::runtime_error("injected failure");
  }
  static void allocate(void** pointer, std::size_t bytes, const char*) {
    step();
    *pointer = new char[bytes];
    buffers.insert(*pointer);
  }
  static void free(void* pointer) noexcept {
    if (buffers.erase(pointer) != 1) std::terminate();
    delete[] static_cast<char*>(pointer);
  }
  static void copy(void* destination, const void* source, std::size_t bytes, const char*) {
    step();
    std::memcpy(destination, source, bytes);
  }
  static void create_event(Event* event, const char*) {
    step();
    *event = new char;
    events.insert(*event);
  }
  static void destroy_event(Event event) noexcept {
    if (events.erase(event) != 1) std::terminate();
    delete static_cast<char*>(event);
  }
};

struct Workspace {
  void* buffers[3]{};
  Workspace() {
    try {
      for (auto& buffer : buffers) Runtime::allocate(&buffer, 16, "workspace");
    } catch (...) {
      for (void* buffer : buffers) if (buffer) Runtime::free(buffer);
      throw;
    }
  }
  ~Workspace() { for (void* buffer : buffers) Runtime::free(buffer); }
};

struct Runner {
  std::shared_ptr<Workspace> workspace;
  std::unique_ptr<gevva::detail::Fp8Resources<Runtime>> resources;
  explicit Runner(std::shared_ptr<Workspace> shared = {}) {
    resources = std::make_unique<gevva::detail::Fp8Resources<Runtime>>(32, 16, 64, [&] {
      workspace = shared ? shared : std::make_shared<Workspace>();
      Runtime::step();  // A checked workspace invariant can also fail.
    });
  }
};

int main() try {
  int failures = 0;
  for (bool shared : {false, true}) {
    Runtime::fail_at = 0;
    auto workspace = shared ? std::make_shared<Workspace>() : std::shared_ptr<Workspace>{};
    const auto baseline = Runtime::buffers.size();
    Runtime::calls = 0;
    { Runner good(workspace); }
    const int steps = Runtime::calls;
    if (Runtime::buffers.size() != baseline || !Runtime::events.empty())
      throw std::runtime_error("successful destruction leaked resources");
    for (int index = 1; index <= steps; ++index) {
      Runtime::calls = 0;
      Runtime::fail_at = index;
      bool threw = false;
      try { Runner failure(workspace); }
      catch (const std::runtime_error&) { threw = true; }
      if (!threw || Runtime::buffers.size() != baseline || !Runtime::events.empty())
        throw std::runtime_error("resource leak after initialization step " + std::to_string(index));
      ++failures;
    }
  }
  if (!Runtime::buffers.empty() || !Runtime::events.empty()) throw std::runtime_error("final resource leak");
  std::cout << "Verified " << failures << " injected allocation/copy/workspace/event failures and normal teardown\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
