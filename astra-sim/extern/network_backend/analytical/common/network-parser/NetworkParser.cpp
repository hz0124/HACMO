/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/NetworkParser.h"
#include <cassert>
#include <iostream>

using namespace NetworkAnalytical;

NetworkParser::NetworkParser(const std::string& path) noexcept
    : dims_count(-1),
      rail_optimized(false),
      switch_dim_index(-1),
      mesh_dim_index(-1) {
    // initialize values
    npus_count_per_dim = {};
    bandwidth_per_dim = {};
    latency_per_dim = {};
    topology_per_dim = {};

    try {
        // load network config file
        const auto network_config = YAML::LoadFile(path);

        // parse network configs
        parse_network_config_yml(network_config);
    } catch (const YAML::BadFile& e) {
        // loading network config file failed
        std::cerr << "[Error] (network/analytical) " << e.what() << std::endl;
        std::exit(-1);
    }
}

int NetworkParser::get_dims_count() const noexcept {
    assert(dims_count > 0);

    return dims_count;
}

std::vector<int> NetworkParser::get_npus_counts_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(npus_count_per_dim.size() == dims_count);

    return npus_count_per_dim;
}

std::vector<Bandwidth> NetworkParser::get_bandwidths_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(bandwidth_per_dim.size() == dims_count);

    return bandwidth_per_dim;
}

std::vector<Latency> NetworkParser::get_latencies_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(latency_per_dim.size() == dims_count);

    return latency_per_dim;
}

std::vector<TopologyBuildingBlock> NetworkParser::get_topologies_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(topology_per_dim.size() == dims_count);

    return topology_per_dim;
}

bool NetworkParser::is_rail_optimized() const noexcept {
    return rail_optimized;
}

int NetworkParser::get_switch_dim_index() const noexcept {
    return switch_dim_index;
}

int NetworkParser::get_mesh_dim_index() const noexcept {
    return mesh_dim_index;
}

std::vector<int> NetworkParser::get_rail_port_nodes() const noexcept {
    return rail_port_nodes;
}

std::vector<double> NetworkParser::get_rail_port_bw_scales() const noexcept {
    return rail_port_bw_scales;
}

void NetworkParser::parse_network_config_yml(const YAML::Node& network_config) noexcept {
    // parse topology_per_dim
    const auto topology_names = parse_vector<std::string>(network_config["topology"]);
    for (const auto& topology_name : topology_names) {
        const auto topology_dim = NetworkParser::parse_topology_name(topology_name);
        topology_per_dim.push_back(topology_dim);
    }

    // set dims_count
    dims_count = static_cast<int>(topology_per_dim.size());

    // parse vector values
    npus_count_per_dim = parse_vector<int>(network_config["npus_count"]);
    bandwidth_per_dim = parse_vector<Bandwidth>(network_config["bandwidth"]);
    latency_per_dim = parse_vector<Latency>(network_config["latency"]);

    // optional rail-optimized config
    if (network_config["rail-optimized"]) {
        rail_optimized = network_config["rail-optimized"].as<bool>();
    }
    if (network_config["switch-dim-index"]) {
        switch_dim_index = network_config["switch-dim-index"].as<int>();
    }
    if (network_config["mesh-dim-index"]) {
        mesh_dim_index = network_config["mesh-dim-index"].as<int>();
    }
    if (network_config["rail-port-nodes"]) {
        rail_port_nodes = parse_vector<int>(network_config["rail-port-nodes"]);
    }
    if (network_config["rail-port-bw-scales"]) {
        rail_port_bw_scales = parse_vector<double>(network_config["rail-port-bw-scales"]);
    }
    if (rail_optimized && rail_port_bw_scales.empty() && !rail_port_nodes.empty()) {
        rail_port_bw_scales.assign(rail_port_nodes.size(), 1.0);
    }

    // check the validity of the parsed network config
    check_validity();
}

TopologyBuildingBlock NetworkParser::parse_topology_name(const std::string& topology_name) noexcept {
    assert(!topology_name.empty());

    if (topology_name == "Ring") {
        return TopologyBuildingBlock::Ring;
    }

    if (topology_name == "FullyConnected") {
        return TopologyBuildingBlock::FullyConnected;
    }

    if (topology_name == "Switch") {
        return TopologyBuildingBlock::Switch;
    }

    if (topology_name == "Mesh") {
        return TopologyBuildingBlock::Mesh;
    }

    // shouldn't reach here
    std::cerr << "[Error] (network/analytical) " << "Topology name " << topology_name << " not supported" << std::endl;
    std::exit(-1);
}

void NetworkParser::check_validity() const noexcept {
    // dims_count should match
    if (dims_count != npus_count_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of npus_count (" << npus_count_per_dim.size()
                  << ") doesn't match with dimensions (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    if (dims_count != bandwidth_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of bandwidth (" << bandwidth_per_dim.size()
                  << ") doesn't match with dims_count (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    if (dims_count != latency_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of latency (" << latency_per_dim.size()
                  << ") doesn't match with dims_count (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    // npus_count should be all positive
    for (const auto& npus_count : npus_count_per_dim) {
        if (npus_count <= 1) {
            std::cerr << "[Error] (network/analytical) " << "npus_count (" << npus_count << ") should be larger than 1"
                      << std::endl;
            std::exit(-1);
        }
    }

    // bandwidths should be all positive
    for (const auto& bandwidth : bandwidth_per_dim) {
        if (bandwidth <= 0) {
            std::cerr << "[Error] (network/analytical) " << "bandwidth (" << bandwidth << ") should be larger than 0"
                      << std::endl;
            std::exit(-1);
        }
    }

    // latency should be non-negative
    for (const auto& latency : latency_per_dim) {
        if (latency < 0) {
            std::cerr << "[Error] (network/analytical) " << "latency (" << latency << ") should be non-negative"
                      << std::endl;
            std::exit(-1);
        }
    }

    if (rail_optimized) {
        if (dims_count != 2) {
            std::cerr << "[Error] (network/analytical) rail-optimized mode currently requires 2 dimensions"
                      << std::endl;
            std::exit(-1);
        }
        if (switch_dim_index < 0 || switch_dim_index >= dims_count || mesh_dim_index < 0 || mesh_dim_index >= dims_count ||
            switch_dim_index == mesh_dim_index) {
            std::cerr << "[Error] (network/analytical) invalid switch-dim-index / mesh-dim-index for rail-optimized mode"
                      << std::endl;
            std::exit(-1);
        }
        if (topology_per_dim[switch_dim_index] != TopologyBuildingBlock::Switch) {
            std::cerr << "[Error] (network/analytical) switch-dim-index should point to Switch topology dimension"
                      << std::endl;
            std::exit(-1);
        }
        if (topology_per_dim[mesh_dim_index] != TopologyBuildingBlock::Mesh) {
            std::cerr << "[Error] (network/analytical) mesh-dim-index should point to Mesh topology dimension"
                      << std::endl;
            std::exit(-1);
        }
        if (rail_port_nodes.empty()) {
            std::cerr << "[Error] (network/analytical) rail-port-nodes should be provided in rail-optimized mode"
                      << std::endl;
            std::exit(-1);
        }
        const auto mesh_npus = npus_count_per_dim[mesh_dim_index];
        for (const auto& node : rail_port_nodes) {
            if (node < 0 || node >= mesh_npus) {
                std::cerr << "[Error] (network/analytical) rail-port-nodes contains invalid mesh local node id: " << node
                          << std::endl;
                std::exit(-1);
            }
        }
        if (rail_port_bw_scales.size() != rail_port_nodes.size()) {
            std::cerr << "[Error] (network/analytical) rail-port-bw-scales length should match rail-port-nodes length"
                      << std::endl;
            std::exit(-1);
        }
        for (const auto& scale : rail_port_bw_scales) {
            if (scale <= 0) {
                std::cerr << "[Error] (network/analytical) rail-port-bw-scales should be positive" << std::endl;
                std::exit(-1);
            }
        }
    }
}
