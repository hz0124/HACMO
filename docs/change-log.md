# Change log

## Conference source preparation — 2026-10-28

- Establish camera-ready-v1 as the source baseline using Zenodo API file
  checksums and the archived SHA256 manifest.
- Rename the three method directories to attention, moe, and interconnect;
  update imports and path references while preserving numerical code.
- Preserve reference tables and their checksums exactly.
- Replace development-workspace documentation with public installation,
  architecture, data, version correspondence, and reproduction instructions.
- Correct old request counts, environment instructions, and Overall provenance.
- Add editable-install metadata, one simulator build entry, and short CI checks.
- Make full preflight reject a missing ASTRA binary and raise the documented
  file-descriptor limit for quick checks.
- Permit ASTRA trace-generation-only runs without a built executable; actual
  simulation still requires it. The existing dry-run test covers this fix.
- Regenerate C++ Chakra bindings using the installed system Protobuf compiler
  on source builds, avoiding dependence on the packaging host's compiler ABI.
- Expand Ramulator2's yaml-cpp, spdlog, and argparse dependencies from the
  verified conference runtime archive and build them without network fetches.
- Recover four omitted CMake helpers from the original upstream version
  commits, after matching all existing dependency files byte-for-byte.

The Zenodo records and original archives remain unchanged.
