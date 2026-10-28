/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_unaware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionUnaware {

/**
 * Implements a 2D square mesh topology on a single basic dimension.
 * The npus_count must be a perfect square (e.g., 16 -> 4x4).
 */
class Mesh final : public BasicTopology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of NPUs (must be perfect square)
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     */
    Mesh(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

  private:
    /**
     * Manhattan distance over the 2D mesh.
     */
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    /// mesh side length for a side x side mesh
    int side;
};

}  // namespace NetworkAnalyticalCongestionUnaware
