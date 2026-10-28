/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_unaware/BasicTopology.h"
#include "congestion_unaware/Topology.h"
#include <memory>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionUnaware {

/**
 * MultiDimTopology implements multi-dimensional network topologies
 * which can be constructed by stacking up multiple BasicTopology instances.
 */
class MultiDimTopology : public Topology {
  public:
    /**
     * Constructor.
     */
    MultiDimTopology() noexcept;

    /**
     * Implement the send method of Topology.
     */
    [[nodiscard]] EventTime send(DeviceId src, DeviceId dest, ChunkSize chunk_size) const noexcept override;

    /**
     * Add a dimension to the multi-dimensional topology.
     *
     * @param topology BasicTopology instance to be added.
     */
    void append_dimension(std::unique_ptr<BasicTopology> basic_topology) noexcept;

    /**
     * Configure rail-optimized inter-wafer routing:
     * - switch_dim_index: dimension index for wafer-level switch
     * - mesh_dim_index: dimension index for wafer-internal mesh
     * - rail_port_nodes: rail index -> mesh local node id
     * - rail_port_bw_scales: rail index -> switch serialization scaling
     */
    void configure_rail_optimized(int switch_dim_index,
                                  int mesh_dim_index,
                                  std::vector<int> rail_port_nodes,
                                  std::vector<double> rail_port_bw_scales) noexcept;

  private:
    /// Each NPU ID can be broken down into multiple dimensions.
    /// for example, if the topology size is [2, 8, 4] and the NPU ID is 31,
    /// then the NPU ID can be broken down into [1, 7, 1].
    using MultiDimAddress = std::vector<DeviceId>;

    /// BasicTopology instances per dimension.
    std::vector<std::unique_ptr<BasicTopology>> topology_per_dim;

    /// whether to enforce edge-port based inter-wafer routing
    bool rail_optimized;

    /// switch dimension index
    int switch_dim_index;

    /// mesh dimension index
    int mesh_dim_index;

    /// rail index -> mesh local node id
    std::vector<int> rail_port_nodes;

    /// rail index -> switch serialization scaling factor
    std::vector<double> rail_port_bw_scales;

    /**
     * Translate the NPU ID into a multi-dimensional address.
     *
     * @param npu_id id of the NPU
     * @return the same NPU in multi-dimensional address representation
     */
    [[nodiscard]] MultiDimAddress translate_address(DeviceId npu_id) const noexcept;

    /**
     * Given src and dest address in multi-dimensional form,
     * return the dimension where the transfer should happen.
     * i.e., the dimension where the src and dest addresses differ.
     *
     * @param src_address src NPU ID in multi-dimensional form
     * @param dest_address dest NPU ID in multi-dimensional form
     * @return the dimension where the transfer should happen
     */
    [[nodiscard]] int get_dim_to_transfer(const MultiDimAddress& src_address,
                                          const MultiDimAddress& dest_address) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionUnaware
