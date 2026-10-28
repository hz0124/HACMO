#include "base/base.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

class NoRefresh : public IRefreshManager, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, NoRefresh, "NoRefresh", "Disable periodic refresh.")

 public:
  void init() override {}
  void tick() override {}
};

}  // namespace Ramulator
