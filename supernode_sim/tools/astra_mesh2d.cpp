/******************************************************************************
This source code is licensed under the MIT license found in the ASTRA
analytical network backend.
*******************************************************************************/

#include "congestion_aware/Mesh2D.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalyticalCongestionAware;

Mesh2D::Mesh2D(const int count,
               const Bandwidth bandwidth,
               const Latency latency) noexcept
    : side(0), Topology() {
    assert(count > 1);
    assert(bandwidth > 0);
    assert(latency >= 0);

    side = static_cast<int>(std::sqrt(static_cast<double>(count)) + 1e-9);
    if (side * side != count) {
        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "Mesh2D requires a square NPU count, got " << count
                  << std::endl;
        std::exit(-1);
    }

    dims_count = 1;
    npus_count = count;
    devices_count = count;
    npus_count_per_dim = {count};
    bandwidth_per_dim = {bandwidth};
    instantiate_devices();

    for (auto r = 0; r < side; ++r) {
        for (auto c = 0; c < side; ++c) {
            const auto here = rank(r, c);
            if (c + 1 < side) {
                connect(here, rank(r, c + 1), bandwidth, latency, true);
            }
            if (r + 1 < side) {
                connect(here, rank(r + 1, c), bandwidth, latency, true);
            }
        }
    }
}

Route Mesh2D::route(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    auto route = Route();
    route.push_back(devices[src]);
    auto r = row(src);
    auto c = col(src);
    const auto target_r = row(dest);
    const auto target_c = col(dest);

    // Deterministic dimension-order routing: X (columns), then Y (rows).
    while (c != target_c) {
        c += (target_c > c) ? 1 : -1;
        route.push_back(devices[rank(r, c)]);
    }
    while (r != target_r) {
        r += (target_r > r) ? 1 : -1;
        route.push_back(devices[rank(r, c)]);
    }
    return route;
}

int Mesh2D::row(const DeviceId id) const noexcept {
    return id / side;
}

int Mesh2D::col(const DeviceId id) const noexcept {
    return id % side;
}

DeviceId Mesh2D::rank(const int r, const int c) const noexcept {
    return r * side + c;
}
