"""Crossbar edge interconnect plugin for SupernodeSim.

This plugin models a crossbar architecture that decouples wafer-edge I/O ports
from die-edge ports. In the baseline rail-only architecture, each I/O port is
hardwired to a specific die edge, creating a bandwidth bottleneck when traffic
is uneven. The crossbar provides non-blocking connectivity between all 4*n
I/O ports and the switch plane, enabling dynamic bandwidth reallocation.

This is a technology-point plugin (TP3), NOT part of supernode_sim's core.
It extends the astra backend's trace generation to support crossbar routing
without modifying the main simulation path.
"""
