#define BOOST_ERROR_CODE_HEADER_ONLY
#include "g4/openai_server.hpp"

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace g4 {
namespace {

using Json = nlohmann::json;
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

std::vector<std::uint8_t> decode_base64(std::string_view source) {
  auto decode = [](unsigned char character) {
    if (character >= 'A' && character <= 'Z') return character - 'A';
    if (character >= 'a' && character <= 'z') return character - 'a' + 26;
    if (character >= '0' && character <= '9') return character - '0' + 52;
    if (character == '+') return 62;
    if (character == '/') return 63;
    if (character == '=') return -2;
    return -1;
  };
  std::vector<std::uint8_t> output;
  output.reserve(source.size() * 3 / 4);
  unsigned accumulator = 0;
  int bits = 0;
  for (const unsigned char character : source) {
    const int value = decode(character);
    if (value == -2) break;
    if (value < 0) throw std::runtime_error("invalid base64 image payload");
    accumulator = (accumulator << 6) | static_cast<unsigned>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output.push_back(static_cast<std::uint8_t>(accumulator >> bits));
      accumulator &= (1U << bits) - 1U;
    }
  }
  return output;
}

struct ParsedRequest {
  std::string system;
  std::string user;
  std::vector<std::uint8_t> image;
  int maximum_tokens{};
  bool stop_on_json_object{};
  std::vector<std::string> expected_json_ids;
  bool stream{};
};

ParsedRequest parse_request(const Json& request) {
  ParsedRequest result;
  result.maximum_tokens = request.value("max_tokens", 8192);
  if (const auto found = request.find("response_format");
      found != request.end() && found->is_object())
    result.stop_on_json_object = found->value("type", "") == "json_object";
  result.stream = request.value("stream", false);
  if (request.value("temperature", 0.0) != 0.0)
    throw std::runtime_error("this greedy runtime requires temperature=0");
  for (const auto& message : request.at("messages")) {
    const auto role = message.at("role").get<std::string>();
    const auto& content = message.at("content");
    auto append_text = [&](std::string& destination, std::string_view text) {
      if (!destination.empty()) destination.push_back('\n');
      destination.append(text);
    };
    if (content.is_string()) {
      if (role == "system" || role == "developer")
        append_text(result.system, content.get_ref<const std::string&>());
      else if (role == "user")
        append_text(result.user, content.get_ref<const std::string&>());
      continue;
    }
    if (!content.is_array()) continue;
    for (const auto& part : content) {
      const auto type = part.value("type", "");
      if (type == "text") {
        auto& destination = (role == "system" || role == "developer")
                                ? result.system : result.user;
        append_text(destination, part.value("text", ""));
      } else if (type == "image_url") {
        if (!result.image.empty())
          throw std::runtime_error("only one image per request is supported");
        const auto url = part.at("image_url").at("url").get<std::string>();
        const auto marker = url.find(";base64,");
        if (!url.starts_with("data:image/") || marker == std::string::npos)
          throw std::runtime_error("image_url must be an inline base64 data URL");
        result.image = decode_base64(std::string_view(url).substr(marker + 8));
      }
    }
  }
  if (result.user.empty()) throw std::runtime_error("request has no user text");
  if (result.stop_on_json_object) {
    const auto prompt_json = Json::parse(result.user, nullptr, false);
    if (!prompt_json.is_discarded() && prompt_json.is_object()) {
      const auto regions = prompt_json.find("input_regions");
      if (regions != prompt_json.end() && regions->is_array())
        for (const auto& region : *regions)
          if (region.is_object() && region.contains("id"))
            result.expected_json_ids.push_back(
                region.at("id").get<std::string>());
    }
  }
  return result;
}

Json completion_json(const GenerationResult& result, std::uint64_t id) {
  const int completion_tokens = static_cast<int>(result.token_ids.size());
  return {{"id", "chatcmpl-g4-" + std::to_string(id)},
          {"object", "chat.completion"},
          {"created", std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count()},
          {"model", "page-vlm"},
          {"choices", Json::array({{{"index", 0},
                                      {"message", {{"role", "assistant"},
                                                   {"content", result.text}}},
                                      {"finish_reason", result.length_limited
                                                            ? "length"
                                                            : "stop"}}})},
          {"usage", {{"prompt_tokens", result.prompt_tokens},
                     {"completion_tokens", completion_tokens},
                     {"total_tokens", result.prompt_tokens + completion_tokens}}},
          {"g4", {{"mtp_cycles", result.cycles.size()},
                   {"accepted_drafts", result.accepted_drafts},
                   {"draft_acceptance", result.draft_acceptance()},
                   {"assistant_ms", result.assistant_milliseconds},
                   {"verifier_ms", result.verifier_milliseconds},
                   {"prefill_us", result.prefill_microseconds},
                   {"vision_us", result.vision_microseconds},
                   {"first_token_ms", result.first_token_milliseconds},
                   {"wall_ms", result.wall_milliseconds}}}};
}

Json error_json(std::string message) {
  return {{"error", {{"message", std::move(message)},
                      {"type", "invalid_request_error"},
                      {"code", nullptr}}}};
}

}  // namespace

struct OpenAiServer::Impl {
  struct Job {
    ParsedRequest request;
    std::promise<Json> response;
    std::uint64_t id{};
    std::atomic_bool completed{};
    std::mutex stream_mutex;
    std::condition_variable stream_ready;
    std::deque<std::string> stream_chunks;
    std::string stream_error;
    bool stream_done{};
    int prompt_tokens{};
    int completion_tokens{};
    bool length_limited{};
  };

  struct Ticket {
    std::shared_ptr<Job> job;
    std::future<Json> future;
  };

  MultimodalRuntime& runtime;
  int maximum_batch;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::shared_ptr<Job>> queue;
  std::atomic<int> open_sessions{};
  bool stopping{};
  int profile_countdown{-1};
  std::thread scheduler;
  std::uint64_t next_id{};

  Impl(MultimodalRuntime& owner, int batch)
      : runtime(owner), maximum_batch(batch),
        profile_countdown(std::getenv("G4_PROFILE_BATCH_INDEX")
                              ? std::atoi(std::getenv("G4_PROFILE_BATCH_INDEX"))
                              : -1),
        scheduler([this] { schedule(); }) {}

  ~Impl() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    ready.notify_all();
    if (scheduler.joinable()) scheduler.join();
  }

  Ticket enqueue(ParsedRequest request) {
    auto job = std::make_shared<Job>();
    job->request = std::move(request);
    auto future = job->response.get_future();
    {
      std::lock_guard lock(mutex);
      job->id = next_id++;
      queue.push_back(job);
    }
    ready.notify_one();
    return {std::move(job), std::move(future)};
  }

  void schedule() {
    for (;;) {
      std::vector<std::shared_ptr<Job>> cohort;
      {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return stopping || !queue.empty(); });
        if (stopping && queue.empty()) return;
        // Give the accept loop five milliseconds to expose concurrent clients.
        // If several sockets are already parsing/uploading, wait for those
        // known requests rather than launching B1 and refilling seven slots.
        // This still halves the old fixed ten-millisecond delay for B1 while
        // covering the connection skew of localmaxxing's eight goroutines.
        ready.wait_for(lock, std::chrono::milliseconds(5), [&] {
          return stopping || static_cast<int>(queue.size()) >= maximum_batch;
        });
        const int expected = std::min(
            maximum_batch,
            std::max(1, open_sessions.load(std::memory_order_relaxed)));
        if (static_cast<int>(queue.size()) < expected)
          ready.wait_for(lock, std::chrono::milliseconds(49), [&] {
            return stopping || static_cast<int>(queue.size()) >= expected;
          });
        while (!queue.empty() &&
               static_cast<int>(cohort.size()) < maximum_batch) {
          cohort.push_back(std::move(queue.front()));
          queue.pop_front();
        }
      }
      std::vector<std::shared_ptr<Job>> admitted = cohort;
      try {
        // Spans and views remain valid because the cohort owns every Job until
        // all responses have been fulfilled below.
        auto make_request = [](const std::shared_ptr<Job>& job) {
          const auto& request = job->request;
          return BatchGenerationRequest{
              request.image, request.system, request.user,
              request.maximum_tokens, request.stop_on_json_object,
              request.expected_json_ids,
              [job](GenerationResult result) {
                if (job->request.stream) {
                  {
                    std::lock_guard lock(job->stream_mutex);
                    job->prompt_tokens = result.prompt_tokens;
                    job->completion_tokens =
                        static_cast<int>(result.token_ids.size());
                    job->length_limited = result.length_limited;
                    job->stream_done = true;
                  }
                  job->stream_ready.notify_one();
                }
                if (!job->completed.exchange(true))
                  job->response.set_value(completion_json(result, job->id));
              },
              [job](std::string piece) {
                if (!job->request.stream || piece.empty()) return;
                {
                  std::lock_guard lock(job->stream_mutex);
                  job->stream_chunks.push_back(std::move(piece));
                }
                job->stream_ready.notify_one();
              }};
        };
        std::vector<BatchGenerationRequest> requests;
        requests.reserve(cohort.size());
        for (const auto& job : cohort) requests.push_back(make_request(job));
        const bool profile_batch = profile_countdown == 0;
        if (profile_countdown >= 0) --profile_countdown;
        if (profile_batch && cudaProfilerStart() != cudaSuccess)
          throw std::runtime_error("cudaProfilerStart failed");
        try {
          // All requests, including text-only B1, use the continuous path.
          // The former scalar shortcut returned one completed string after
          // generation and therefore made an SSE response look like a single
          // buffered token burst. It also prevented an arriving request from
          // refilling the active batch.
          runtime.generate_continuous(
              requests, [&, this]() -> std::optional<BatchGenerationRequest> {
                std::shared_ptr<Job> job;
                {
                  std::lock_guard lock(mutex);
                  // Never stall live decode rows waiting for a replacement.
                  // The runtime polls vacant slots again at every MTP
                  // boundary, after the completed response has had time to
                  // cross the socket and release the client's semaphore.
                  if (queue.empty()) return std::nullopt;
                  job = std::move(queue.front());
                  queue.pop_front();
                }
                admitted.push_back(job);
                return make_request(job);
              });
        } catch (...) {
          if (profile_batch) cudaProfilerStop();
          throw;
        }
        if (profile_batch && cudaProfilerStop() != cudaSuccess)
          throw std::runtime_error("cudaProfilerStop failed");
      } catch (const std::exception& error) {
        // Preserve the first failure before later requests encounter a poisoned
        // CUDA context. HTTP clients often retain only the 400 status string.
        std::cerr << "g4 serving wave failed: " << error.what()
                  << " (CUDA last error: "
                  << cudaGetErrorString(cudaPeekAtLastError()) << ")\n";
        for (const auto& job : admitted) {
          if (job->request.stream) {
            {
              std::lock_guard lock(job->stream_mutex);
              job->stream_error = error.what();
              job->stream_done = true;
            }
            job->stream_ready.notify_one();
          }
          if (!job->completed.exchange(true))
            job->response.set_value(error_json(error.what()));
        }
      }
    }
  }

  static std::string sse_chunk(const Job& job, Json delta,
                               const char* finish_reason = nullptr) {
    Json choice{{"index", 0}, {"delta", std::move(delta)}};
    choice["finish_reason"] = finish_reason ? Json(finish_reason) : Json(nullptr);
    Json payload{{"id", "chatcmpl-g4-" + std::to_string(job.id)},
                 {"object", "chat.completion.chunk"},
                 {"created", std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::system_clock::now()
                                     .time_since_epoch()).count()},
                 {"model", "page-vlm"},
                 {"choices", Json::array({std::move(choice)})}};
    if (finish_reason)
      payload["usage"] = {{"prompt_tokens", job.prompt_tokens},
                          {"completion_tokens", job.completion_tokens},
                          {"total_tokens", job.prompt_tokens +
                                               job.completion_tokens}};
    return "data: " + payload.dump() + "\n\n";
  }

  void stream_response(tcp::socket& socket, const std::shared_ptr<Job>& job,
                       unsigned version, boost::system::error_code& error) {
    http::response<http::empty_body> header{http::status::ok, version};
    header.keep_alive(false);
    header.set(http::field::server, "g4");
    header.set(http::field::content_type, "text/event-stream");
    header.set(http::field::cache_control, "no-cache");
    header.chunked(true);
    http::response_serializer<http::empty_body> serializer{header};
    http::write_header(socket, serializer, error);
    if (error) return;
    auto write_event = [&](const std::string& event) {
      asio::write(socket, http::make_chunk(asio::buffer(event)), error);
      return !error;
    };
    if (!write_event(sse_chunk(*job, {{"role", "assistant"}}))) return;
    for (;;) {
      std::deque<std::string> pieces;
      bool done = false;
      std::string failure;
      {
        std::unique_lock lock(job->stream_mutex);
        job->stream_ready.wait(lock, [&] {
          return job->stream_done || !job->stream_chunks.empty();
        });
        pieces.swap(job->stream_chunks);
        done = job->stream_done;
        failure = job->stream_error;
      }
      for (auto& piece : pieces)
        if (!write_event(sse_chunk(*job, {{"content", std::move(piece)}})))
          return;
      if (!done) continue;
      if (!failure.empty()) {
        if (!write_event("data: " + error_json(std::move(failure)).dump() +
                         "\n\n"))
          return;
      } else if (!write_event(sse_chunk(
                     *job, Json::object(),
                     job->length_limited ? "length" : "stop"))) {
        return;
      }
      if (!write_event("data: [DONE]\n\n")) return;
      asio::write(socket, http::make_chunk_last(), error);
      return;
    }
  }

  void session(tcp::socket socket) {
    struct SessionCount {
      Impl& owner;
      explicit SessionCount(Impl& value) : owner(value) {
        owner.open_sessions.fetch_add(1, std::memory_order_relaxed);
        owner.ready.notify_one();
      }
      ~SessionCount() {
        owner.open_sessions.fetch_sub(1, std::memory_order_relaxed);
        owner.ready.notify_one();
      }
    } session_count(*this);
    beast::flat_buffer buffer;
    http::request<http::string_body> request;
    http::request_parser<http::string_body> parser;
    parser.body_limit(64ULL << 20);
    boost::system::error_code error;
    http::read(socket, buffer, parser, error);
    if (error) return;
    request = parser.release();
    http::response<http::string_body> response;
    response.version(request.version());
    response.keep_alive(false);
    response.set(http::field::server, "g4");
    response.set(http::field::content_type, "application/json");
    try {
      if (request.method() == http::verb::get && request.target() == "/health") {
        response.result(http::status::ok);
        response.body() = Json({{"status", "ok"}}).dump();
      } else if (request.method() == http::verb::get &&
                 request.target() == "/v1/models") {
        response.result(http::status::ok);
        response.body() = Json({{"object", "list"}, {"data", Json::array({
            {{"id", "page-vlm"}, {"object", "model"}, {"owned_by", "g4"}}})}}).dump();
      } else if (request.method() == http::verb::post &&
                 request.target() == "/v1/chat/completions") {
        auto parsed = parse_request(Json::parse(request.body()));
        const bool stream = parsed.stream;
        auto ticket = enqueue(std::move(parsed));
        if (stream) {
          stream_response(socket, ticket.job, request.version(), error);
          socket.shutdown(tcp::socket::shutdown_send, error);
          return;
        }
        auto payload = ticket.future.get();
        const bool failed = payload.contains("error");
        response.result(failed ? http::status::bad_request : http::status::ok);
        response.body() = payload.dump();
      } else {
        response.result(http::status::not_found);
        response.body() = error_json("route not found").dump();
      }
    } catch (const std::exception& exception) {
      response.result(http::status::bad_request);
      response.body() = error_json(exception.what()).dump();
    }
    response.prepare_payload();
    http::write(socket, response, error);
    socket.shutdown(tcp::socket::shutdown_send, error);
  }
};

OpenAiServer::OpenAiServer(MultimodalRuntime& runtime, int maximum_batch)
    : impl_(new Impl(runtime, maximum_batch)) {
  if (maximum_batch < 1 || maximum_batch > 8)
    throw std::runtime_error("maximum HTTP batch must be in [1, 8]");
}

OpenAiServer::~OpenAiServer() { delete impl_; }

[[noreturn]] void OpenAiServer::run(std::uint16_t port) {
  asio::io_context context(1);
  tcp::acceptor acceptor(context, {asio::ip::make_address("127.0.0.1"), port});
  for (;;) {
    tcp::socket socket(context);
    acceptor.accept(socket);
    std::thread([impl = impl_, socket = std::move(socket)]() mutable {
      impl->session(std::move(socket));
    }).detach();
  }
}

}  // namespace g4
