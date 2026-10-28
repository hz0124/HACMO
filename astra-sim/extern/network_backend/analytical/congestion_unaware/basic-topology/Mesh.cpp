/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/Mesh.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

Mesh::Mesh(const int npus_count, const Bandwidth bandwidth, const Latency latency) noexcept
    : side(0),
      BasicTopology(npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    const auto side_d = std::sqrt(static_cast<double>(npus_count));
    side = static_cast<int>(side_d + 1e-9);
    if (side * side != npus_count) {
        std::cerr << "[Error] (network/analytical/congestion_unaware) "
                  << "Mesh topology requires perfect-square npus_count, got " << npus_count << std::endl;
        std::exit(-1);
    }

    basic_topology_type = TopologyBuildingBlock::Mesh;
}

int Mesh::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const auto src_x = src / side;
    const auto src_y = src % side;
    const auto dst_x = dest / side;
    const auto dst_y = dest % side;
    return std::abs(src_x - dst_x) + std::abs(src_y - dst_y);
}
