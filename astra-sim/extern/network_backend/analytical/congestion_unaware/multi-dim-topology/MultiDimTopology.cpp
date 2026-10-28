/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/MultiDimTopology.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

MultiDimTopology::MultiDimTopology() noexcept
    : rail_optimized(false),
      switch_dim_index(-1),
      mesh_dim_index(-1),
      Topology() {
    // initialize values
    topology_per_dim.clear();
    npus_count_per_dim = {};

    // initialize topology shape
    npus_count = 1;
    dims_count = 0;
}

EventTime MultiDimTopology::send(const DeviceId src, const DeviceId dest, const ChunkSize chunk_size) const noexcept {
    // translate src and dest to multi-dim address
    const auto src_address = translate_address(src);
    const auto dest_address = translate_address(dest);

    if (rail_optimized) {
        const auto src_wafer = src_address[switch_dim_index];
        const auto dst_wafer = dest_address[switch_dim_index];
        const auto src_local = src_address[mesh_dim_index];
        const auto dst_local = dest_address[mesh_dim_index];

        auto* const switch_topology = topology_per_dim[switch_dim_index].get();
        auto* const mesh_topology = topology_per_dim[mesh_dim_index].get();

        // Same wafer: pure intra-wafer mesh communication.
        if (src_wafer == dst_wafer) {
            if (src_local == dst_local) {
                return 0;
            }
            return mesh_topology->send(src_local, dst_local, chunk_size);
        }

        // Cross-wafer:
        // src_local -> nearest source-side edge port (mesh)
        // switch transfer
        // nearest destination-side edge port -> dst_local (mesh)
        //
        // The source and destination ports are selected independently.  A real
        // switch plane can route between arbitrary edge I/O ports; forcing the
        // same local edge port on both wafers overestimates mesh distance and
        // hides destination-side port locality.
        const auto unit_mesh_delay = [mesh_topology](const DeviceId a, const DeviceId b) -> EventTime {
            return a == b ? 0 : mesh_topology->send(a, b, 1);
        };
        auto best_src_port = rail_port_nodes.front();
        auto best_src_hops = unit_mesh_delay(src_local, best_src_port);
        auto best_src_scale = rail_port_bw_scales.front();
        auto best_dst_port = rail_port_nodes.front();
        auto best_dst_hops = unit_mesh_delay(best_dst_port, dst_local);
        auto best_dst_scale = rail_port_bw_scales.front();
        for (size_t rail = 1; rail < rail_port_nodes.size(); rail++) {
            const auto port_node = rail_port_nodes[rail];
            const auto bw_scale = rail_port_bw_scales[rail];
            const auto src_hops = unit_mesh_delay(src_local, port_node);
            if (src_hops < best_src_hops || (src_hops == best_src_hops && bw_scale > best_src_scale)) {
                best_src_port = port_node;
                best_src_hops = src_hops;
                best_src_scale = bw_scale;
            }
            const auto dst_hops = unit_mesh_delay(port_node, dst_local);
            if (dst_hops < best_dst_hops || (dst_hops == best_dst_hops && bw_scale > best_dst_scale)) {
                best_dst_port = port_node;
                best_dst_hops = dst_hops;
                best_dst_scale = bw_scale;
            }
        }

        EventTime src_mesh = 0;
        if (src_local != best_src_port) {
            src_mesh = mesh_topology->send(src_local, best_src_port, chunk_size);
        }
        EventTime dst_mesh = 0;
        if (dst_local != best_dst_port) {
            dst_mesh = mesh_topology->send(best_dst_port, dst_local, chunk_size);
        }

        const auto bw_scale = std::min(best_src_scale, best_dst_scale);
        const auto scaled_chunk =
            static_cast<ChunkSize>(std::ceil(static_cast<double>(chunk_size) / bw_scale));
        const auto switch_delay = switch_topology->send(src_wafer, dst_wafer, scaled_chunk);
        return src_mesh + switch_delay + dst_mesh;
    }

    // get dim to transfer
    const auto dim_to_transfer = get_dim_to_transfer(src_address, dest_address);

    // prepare localized topology and address info
    auto* const topology = topology_per_dim[dim_to_transfer].get();
    const auto src_local_id = src_address[dim_to_transfer];
    const auto dest_local_id = dest_address[dim_to_transfer];

    // run localized communication
    const auto comms_delay = topology->send(src_local_id, dest_local_id, chunk_size);

    // return communication delay
    return comms_delay;
}

void MultiDimTopology::configure_rail_optimized(const int switch_dim,
                                                const int mesh_dim,
                                                std::vector<int> rail_nodes,
                                                std::vector<double> rail_scales) noexcept {
    switch_dim_index = switch_dim;
    mesh_dim_index = mesh_dim;
    rail_port_nodes = std::move(rail_nodes);
    rail_port_bw_scales = std::move(rail_scales);
    rail_optimized = true;
}

void MultiDimTopology::append_dimension(std::unique_ptr<BasicTopology> topology) noexcept {
    // increment dims_count
    dims_count++;

    // increase npus_count
    const auto topology_size = topology->get_npus_count();
    npus_count *= topology_size;

    // append bandwidth
    const auto bandwidth = topology->get_bandwidth_per_dim()[0];
    bandwidth_per_dim.push_back(bandwidth);

    // push back topology and npus_count
    topology_per_dim.push_back(std::move(topology));
    npus_count_per_dim.push_back(topology_size);
}

MultiDimTopology::MultiDimAddress MultiDimTopology::translate_address(const DeviceId npu_id) const noexcept {
    // If units-count if [2, 8, 4], and the given id is 47, then the id should be
    // 47 // 16 = 2, leftover = 47 % 16 = 15
    // 15 // 2 = 7, leftover = 15 % 2 = 1
    // 1 // 1 = 1, leftover = 0
    // therefore the address is [1, 7, 2]

    // create empty address
    auto multi_dim_address = MultiDimAddress();
    for (auto i = 0; i < dims_count; i++) {
        multi_dim_address.push_back(-1);
    }

    auto leftover = npu_id;
    auto denominator = npus_count;

    for (auto dim = dims_count - 1; dim >= 0; dim--) {
        // change denominator
        denominator /= npus_count_per_dim[dim];

        // get and update address
        const auto quotient = leftover / denominator;
        leftover %= denominator;

        // update address
        multi_dim_address[dim] = quotient;
    }

    // check address translation
    for (auto i = 0; i < dims_count; i++) {
        assert(0 <= multi_dim_address[i]);
        assert(multi_dim_address[i] < npus_count_per_dim[i]);
    }

    // return retrieved address
    return multi_dim_address;
}

int MultiDimTopology::get_dim_to_transfer(const MultiDimAddress& src_address,
                                          const MultiDimAddress& dest_address) const noexcept {
    for (auto dim = 0; dim < dims_count; dim++) {
        // check the dim that has different address
        if (src_address[dim] != dest_address[dim]) {
            return dim;
        }
    }

    // shouldn't reach here
    std::cerr << "[Error] (network/analytical/congestion_unaware): " << "src and dest have the same address"
              << std::endl;
    std::exit(-1);
}
