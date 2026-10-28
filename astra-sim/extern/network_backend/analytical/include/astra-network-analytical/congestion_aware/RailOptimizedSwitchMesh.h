/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Topology.h"
#include <vector>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Congestion-aware [Switch, Mesh] topology for wafer-scale supernodes.
 *
 * NPU ids are row-major within each wafer:
 *   global_id = wafer_id * dies_per_wafer + local_mesh_id
 *
 * Same-wafer traffic is routed on the 2-D mesh. Cross-wafer traffic is routed
 * from the source die to the nearest source-side edge I/O port, through a
 * shared switch fabric, then from the nearest destination-side edge I/O port to
 * the destination die. Each edge port has its own queued links to/from the
 * switch, so bandwidth contention on hot I/O ports is modeled by Link queues.
 */
class RailOptimizedSwitchMesh final : public Topology {
  public:
    RailOptimizedSwitchMesh(int num_wafers,
                            int dies_per_wafer,
                            Bandwidth switch_bandwidth,
                            Bandwidth mesh_bandwidth,
                            Latency switch_latency,
                            Latency mesh_latency,
                            std::vector<int> rail_port_nodes,
                            std::vector<double> rail_port_bw_scales) noexcept;

    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

  private:
    int num_wafers;
    int dies_per_wafer;
    int mesh_side;
    DeviceId switch_id;

    std::vector<int> rail_port_nodes;
    std::vector<double> rail_port_bw_scales;

    [[nodiscard]] int wafer(DeviceId id) const noexcept;
    [[nodiscard]] int local(DeviceId id) const noexcept;
    [[nodiscard]] DeviceId global(int wafer_id, int local_id) const noexcept;
    [[nodiscard]] int row(int local_id) const noexcept;
    [[nodiscard]] int col(int local_id) const noexcept;
    [[nodiscard]] int rc_to_local(int r, int c) const noexcept;
    [[nodiscard]] int mesh_hops(int local_src, int local_dst) const noexcept;
    [[nodiscard]] int nearest_port(int local_id) const noexcept;

    void append_mesh_route(Route& route, int wafer_id, int local_src, int local_dst) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
