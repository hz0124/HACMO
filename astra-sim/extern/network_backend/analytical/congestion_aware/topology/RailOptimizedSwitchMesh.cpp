/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/RailOptimizedSwitchMesh.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>

using namespace NetworkAnalyticalCongestionAware;

RailOptimizedSwitchMesh::RailOptimizedSwitchMesh(const int num_wafers,
                                                 const int dies_per_wafer,
                                                 const Bandwidth switch_bandwidth,
                                                 const Bandwidth mesh_bandwidth,
                                                 const Latency switch_latency,
                                                 const Latency mesh_latency,
                                                 std::vector<int> rail_port_nodes,
                                                 std::vector<double> rail_port_bw_scales) noexcept
    : num_wafers(num_wafers),
      dies_per_wafer(dies_per_wafer),
      mesh_side(0),
      switch_id(-1),
      rail_port_nodes(std::move(rail_port_nodes)),
      rail_port_bw_scales(std::move(rail_port_bw_scales)),
      Topology() {
    assert(num_wafers > 1);
    assert(dies_per_wafer > 1);
    assert(switch_bandwidth > 0);
    assert(mesh_bandwidth > 0);
    assert(switch_latency >= 0);
    assert(mesh_latency >= 0);
    assert(!this->rail_port_nodes.empty());
    assert(this->rail_port_nodes.size() == this->rail_port_bw_scales.size());

    const auto side_d = std::sqrt(static_cast<double>(dies_per_wafer));
    mesh_side = static_cast<int>(side_d + 1e-9);
    if (mesh_side * mesh_side != dies_per_wafer) {
        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "RailOptimizedSwitchMesh requires a square mesh, got dies_per_wafer="
                  << dies_per_wafer << std::endl;
        std::exit(-1);
    }

    dims_count = 2;
    npus_count = num_wafers * dies_per_wafer;
    devices_count = npus_count + 1;
    switch_id = npus_count;
    npus_count_per_dim = {num_wafers, dies_per_wafer};
    bandwidth_per_dim = {switch_bandwidth, mesh_bandwidth};

    instantiate_devices();

    for (auto w = 0; w < num_wafers; w++) {
        for (auto r = 0; r < mesh_side; r++) {
            for (auto c = 0; c < mesh_side; c++) {
                const auto here = global(w, rc_to_local(r, c));
                if (r + 1 < mesh_side) {
                    connect(here, global(w, rc_to_local(r + 1, c)), mesh_bandwidth, mesh_latency, true);
                }
                if (c + 1 < mesh_side) {
                    connect(here, global(w, rc_to_local(r, c + 1)), mesh_bandwidth, mesh_latency, true);
                }
            }
        }
    }

    // Multiple edge directions may map to the same corner die. Device::connect
    // supports one link per endpoint pair, so combine duplicate port entries by
    // increasing the effective bandwidth of that die's rail link.
    auto scale_by_port = std::map<int, double>();
    for (size_t i = 0; i < this->rail_port_nodes.size(); i++) {
        const auto port = this->rail_port_nodes[i];
        const auto scale = this->rail_port_bw_scales[i];
        assert(0 <= port && port < dies_per_wafer);
        assert(scale > 0);
        scale_by_port[port] += scale;
    }

    this->rail_port_nodes.clear();
    this->rail_port_bw_scales.clear();
    for (const auto& [port, scale] : scale_by_port) {
        this->rail_port_nodes.push_back(port);
        this->rail_port_bw_scales.push_back(scale);
        for (auto w = 0; w < num_wafers; w++) {
            connect(global(w, port), switch_id, switch_bandwidth * scale, switch_latency, true);
        }
    }
}

Route RailOptimizedSwitchMesh::route(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const auto src_w = wafer(src);
    const auto dst_w = wafer(dest);
    const auto src_l = local(src);
    const auto dst_l = local(dest);

    auto route = Route();
    route.push_back(devices[src]);

    if (src_w == dst_w) {
        append_mesh_route(route, src_w, src_l, dst_l);
        return route;
    }

    const auto src_port = nearest_port(src_l);
    const auto dst_port = nearest_port(dst_l);

    append_mesh_route(route, src_w, src_l, src_port);
    if (route.back()->get_id() != switch_id) {
        route.push_back(devices[switch_id]);
    }
    const auto dst_port_global = global(dst_w, dst_port);
    if (route.back()->get_id() != dst_port_global) {
        route.push_back(devices[dst_port_global]);
    }
    append_mesh_route(route, dst_w, dst_port, dst_l);
    return route;
}

int RailOptimizedSwitchMesh::wafer(const DeviceId id) const noexcept {
    return id / dies_per_wafer;
}

int RailOptimizedSwitchMesh::local(const DeviceId id) const noexcept {
    return id % dies_per_wafer;
}

DeviceId RailOptimizedSwitchMesh::global(const int wafer_id, const int local_id) const noexcept {
    return wafer_id * dies_per_wafer + local_id;
}

int RailOptimizedSwitchMesh::row(const int local_id) const noexcept {
    return local_id / mesh_side;
}

int RailOptimizedSwitchMesh::col(const int local_id) const noexcept {
    return local_id % mesh_side;
}

int RailOptimizedSwitchMesh::rc_to_local(const int r, const int c) const noexcept {
    return r * mesh_side + c;
}

int RailOptimizedSwitchMesh::mesh_hops(const int local_src, const int local_dst) const noexcept {
    return std::abs(row(local_src) - row(local_dst)) + std::abs(col(local_src) - col(local_dst));
}

int RailOptimizedSwitchMesh::nearest_port(const int local_id) const noexcept {
    auto best_port = rail_port_nodes.front();
    auto best_hops = mesh_hops(local_id, best_port);
    auto best_scale = rail_port_bw_scales.front();
    for (size_t i = 1; i < rail_port_nodes.size(); i++) {
        const auto port = rail_port_nodes[i];
        const auto hops = mesh_hops(local_id, port);
        const auto scale = rail_port_bw_scales[i];
        if (hops < best_hops || (hops == best_hops && scale > best_scale) ||
            (hops == best_hops && scale == best_scale && port < best_port)) {
            best_port = port;
            best_hops = hops;
            best_scale = scale;
        }
    }
    return best_port;
}

void RailOptimizedSwitchMesh::append_mesh_route(Route& route,
                                                const int wafer_id,
                                                const int local_src,
                                                const int local_dst) const noexcept {
    if (local_src == local_dst) {
        return;
    }
    auto r = row(local_src);
    auto c = col(local_src);
    const auto target_r = row(local_dst);
    const auto target_c = col(local_dst);

    while (r != target_r) {
        r += (target_r > r) ? 1 : -1;
        route.push_back(devices[global(wafer_id, rc_to_local(r, c))]);
    }
    while (c != target_c) {
        c += (target_c > c) ? 1 : -1;
        route.push_back(devices[global(wafer_id, rc_to_local(r, c))]);
    }
}
