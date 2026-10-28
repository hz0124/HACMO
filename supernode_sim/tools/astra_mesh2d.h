/******************************************************************************
This source code is licensed under the MIT license found in the ASTRA
analytical network backend.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Topology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/** Congestion-aware square 2-D mesh with deterministic X-then-Y routing. */
class Mesh2D final : public Topology {
  public:
    Mesh2D(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

  private:
    int side;

    [[nodiscard]] int row(DeviceId id) const noexcept;
    [[nodiscard]] int col(DeviceId id) const noexcept;
    [[nodiscard]] DeviceId rank(int row, int col) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
