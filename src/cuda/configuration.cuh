// Private implementation fragment; included only by src/gpu.cu.
int fused_norm_threads(const char* specific = nullptr,
                       int default_threads = 512) {
  const char* value = specific ? std::getenv(specific) : nullptr;
  if (!value) value = std::getenv("GEVVA_FUSED_NORM_THREADS");
  if (!value) return default_threads;
  const int parsed = std::atoi(value);
  if (parsed != 128 && parsed != 256 && parsed != 512)
    throw std::runtime_error(
        std::string(specific ? specific : "GEVVA_FUSED_NORM_THREADS") +
        " must be 128, 256, or 512");
  return parsed;
}

