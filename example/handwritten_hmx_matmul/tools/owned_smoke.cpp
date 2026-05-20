#include "handwritten_hmx_matmul.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string JsonEscape(const std::string &value) {
  std::ostringstream os;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << ch;
        break;
    }
  }
  return os.str();
}

std::vector<unsigned char> ReadFile(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input: " + path.string());
  }
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
}

void WriteFile(const fs::path &path, const std::vector<unsigned char> &data) {
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output: " + path.string());
  }
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size()));
}

std::map<std::string, std::string> ParseArgs(int argc, char **argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::runtime_error("expected --key value arguments");
    }
    args[key.substr(2)] = argv[++i];
  }
  return args;
}

std::string Required(const std::map<std::string, std::string> &args,
                     const std::string &key) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    throw std::runtime_error("missing --" + key);
  }
  return it->second;
}

uint32_t RequiredU32(const std::map<std::string, std::string> &args,
                     const std::string &key) {
  return static_cast<uint32_t>(std::stoul(Required(args, key)));
}

HmStatus RunFamily(const std::string &family, const HmPreparedRun *run) {
  if (family == "u8i8") return hm_u8i8_run_prepared(run);
  if (family == "w4a8") return hm_w4a8_run_prepared(run);
  if (family == "w8a16") return hm_w8a16_run_prepared(run);
  if (family == "w4a16") return hm_w4a16_run_prepared(run);
  if (family == "w16a16") return hm_w16a16_run_prepared(run);
  return HM_STATUS_UNSUPPORTED;
}

void WriteOwnedRunJson(const fs::path &path,
                       const std::map<std::string, std::string> &args,
                       HmStatus status,
                       long long runtime_us) {
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write owned_run.json: " + path.string());
  }
  out << "{\n";
  out << "  \"schema\": \"handwritten_hmx_matmul_owned_run.v1\",\n";
  const std::string runtime_kind =
      args.count("runtime-kind") ? args.at("runtime-kind") : "host_smoke";
  const bool device_execution =
      args.count("device-execution") &&
      (args.at("device-execution") == "1" || args.at("device-execution") == "true");
  out << "  \"runtime_kind\": \"" << JsonEscape(runtime_kind) << "\",\n";
  out << "  \"device_execution\": " << (device_execution ? "true" : "false")
      << ",\n";
  out << "  \"qnn_used\": false,\n";
  out << "  \"compute_backend\": \"copy_smoke\",\n";
  out << "  \"hmx_body_entered\": false,\n";
  out << "  \"output_exactness_status\": \"not_checked_copy_smoke\",\n";
  out << "  \"accepted_for_milestone1_device_gate\": "
      << (device_execution && status == HM_STATUS_OK ? "true" : "false") << ",\n";
  out << "  \"accepted_for_milestone4_compute_gate\": false,\n";
  out << "  \"family\": \"" << JsonEscape(args.at("family")) << "\",\n";
  out << "  \"shape_mkn\": [" << args.at("m") << ", " << args.at("k") << ", "
      << args.at("n") << "],\n";
  out << "  \"chain\": " << args.at("chain") << ",\n";
  out << "  \"status\": \"" << hm_status_string(status) << "\",\n";
  out << "  \"runtime_us\": " << runtime_us << ",\n";
  out << "  \"preparation_included\": false,\n";
  out << "  \"command\": [\n";
  const char *keys[] = {"family", "m", "k", "n", "chain", "runtime-kind",
                        "device-execution", "activation", "packed-weight",
                        "folded-bias", "control", "extra-control",
                        "activation-table", "output-table", "descriptor",
                        "mask-control", "output", "output-bytes",
                        "owned-run-json"};
  for (size_t i = 0; i < std::size(keys); ++i) {
    const std::string key = keys[i];
    auto it = args.find(key);
    if (it == args.end()) continue;
    out << "    \"--" << JsonEscape(key) << "\", \""
        << JsonEscape(it->second) << "\"";
    out << (i + 1 == std::size(keys) ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"forbidden_qnn_tools_observed\": []\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    auto args = ParseArgs(argc, argv);
    const std::string family = Required(args, "family");
    const size_t output_bytes = static_cast<size_t>(
        std::stoull(Required(args, "output-bytes")));

    std::vector<unsigned char> activation = ReadFile(Required(args, "activation"));
    std::vector<unsigned char> packed_weight =
        ReadFile(Required(args, "packed-weight"));
    std::vector<unsigned char> folded_bias =
        ReadFile(Required(args, "folded-bias"));
    std::vector<unsigned char> control = ReadFile(Required(args, "control"));
    std::vector<unsigned char> extra_control =
        ReadFile(Required(args, "extra-control"));
    std::vector<unsigned char> activation_table =
        ReadFile(Required(args, "activation-table"));
    std::vector<unsigned char> output_table =
        ReadFile(Required(args, "output-table"));
    std::vector<unsigned char> descriptor =
        ReadFile(Required(args, "descriptor"));
    std::vector<unsigned char> mask_control =
        ReadFile(Required(args, "mask-control"));
    std::vector<unsigned char> scratch(2048, 0);
    std::vector<unsigned char> output(output_bytes, 0);

    HmPreparedRun run = {};
    run.m = RequiredU32(args, "m");
    run.k = RequiredU32(args, "k");
    run.n = RequiredU32(args, "n");
    run.chain = RequiredU32(args, "chain");
    run.activation = {activation.data(), activation.size()};
    run.packed_weight = {packed_weight.data(), packed_weight.size()};
    run.folded_bias = {folded_bias.data(), folded_bias.size()};
    run.control = {control.data(), control.size()};
    run.extra_control = {extra_control.data(), extra_control.size()};
    run.activation_table = {activation_table.data(), activation_table.size()};
    run.output_table = {output_table.data(), output_table.size()};
    run.descriptor = {descriptor.data(), descriptor.size()};
    run.mask_control = {mask_control.data(), mask_control.size()};
    run.scratch = {scratch.data(), scratch.size()};
    run.output = {output.data(), output.size()};

    const auto t0 = std::chrono::steady_clock::now();
    HmStatus status = RunFamily(family, &run);
    const auto t1 = std::chrono::steady_clock::now();
    const auto runtime_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    WriteFile(Required(args, "output"), output);
    WriteOwnedRunJson(Required(args, "owned-run-json"), args, status, runtime_us);
    if (status != HM_STATUS_OK) {
      std::cerr << "owned smoke failed: " << hm_status_string(status) << "\n";
      return static_cast<int>(status);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "owned_smoke: " << e.what() << "\n";
    return 1;
  }
}
