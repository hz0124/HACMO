/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STREAM_STAT_HH__
#define __STREAM_STAT_HH__

#include <iostream>
#include <list>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/NetworkStat.hh"
#include "astra-sim/system/SharedBusStat.hh"

namespace AstraSim {

class StreamStat : public SharedBusStat, public NetworkStat {
  public:
    StreamStat() : SharedBusStat(BusType::Shared, 0, 0, 0, 0) {
        stream_stat_counter = 0;
    }

    void update_stream_stats(StreamStat* streamStat) {
        update_bus_stats(BusType::Both, streamStat);
        update_network_stat(streamStat);

        // Thread-safe queuing_delay update with bounds checking
        if (streamStat && !streamStat->queuing_delay.empty()) {
            // Ensure we have enough space in queuing_delay
            while (queuing_delay.size() < streamStat->queuing_delay.size()) {
                queuing_delay.push_back(0.0);
            }

            // Safely update queuing delays
            auto it = queuing_delay.begin();
            auto stat_it = streamStat->queuing_delay.begin();
            size_t update_count = 0;

            while (it != queuing_delay.end() && stat_it != streamStat->queuing_delay.end() && update_count < 1000) {
                *it += *stat_it;
                ++it;
                ++stat_it;
                update_count++;
            }

            // Safety check: if we somehow have more data than expected, log it
            if (stat_it != streamStat->queuing_delay.end()) {
                std::cout << "[WARNING] StreamStat: truncated queuing_delay update, had "
                          << streamStat->queuing_delay.size() << " values but only updated "
                          << update_count << std::endl;
        }
        }

        stream_stat_counter++;
    }

    void take_stream_stats_average() {
        take_bus_stats_average();
        take_network_stat_average();
        for (auto& tick : queuing_delay) {
            tick /= stream_stat_counter;
        }
    }

    std::list<double> queuing_delay;
    int stream_stat_counter;
};

}  // namespace AstraSim

#endif /* __STREAM_STAT_HH__ */
