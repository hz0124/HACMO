# Contributing

Use an editable installation from the checkout and run `bash ae/run_quick.sh`
before proposing a change. For simulator changes, build with
`bash scripts/build_simulators.sh` and run the affected full workflow stages.
Include commands, dependency versions, and numerical comparison results in the
pull request. Small documentation changes need link and command review.

Keep reference tables immutable. New experimental settings should have their
own explicit configs and output directories. Distinguish simulation results,
analytical approximations, measurements, and archived inputs in documentation.
Preserve all third-party notices. Generated data and credentials must stay out
of commits. Please report bugs through GitHub Issues with a minimal command,
config, platform, Python version, and relevant log excerpt.
