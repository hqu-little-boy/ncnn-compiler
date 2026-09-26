#include "CompileSession.hpp"

namespace ncnn_compile {

int CompileSession::run(int argc, char** argv) {
  if (int status = parseArguments(argc, argv)) {
    return status;
  }
  if (int status = resolveTools(argv)) {
    return status;
  }
  if (int status = resolveTarget()) {
    return status;
  }
  if (int status = declareArtifacts()) {
    return status;
  }
  if (int status = runPipeline()) {
    return status;
  }
  if (int status = emitABI()) {
    return status;
  }
  if (int status = linkAndAudit()) {
    return status;
  }
  if (int status = verifyExecution()) {
    return status;
  }
  if (int status = publishOutputs()) {
    return status;
  }
  return 0;
}

}  // namespace ncnn_compile
