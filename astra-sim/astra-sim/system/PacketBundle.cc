/*
 * Copyright (c) 2022-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PacketBundle.hh"
#include <iostream>
#include "Sys.hh"
#include "StreamBaseline.hh"
#include "MemBus.hh"
#include "MyPacket.hh"

using namespace AstraSim;

PacketBundle::PacketBundle(
    Sys* sys,
                           BaseStream* stream,
                           std::list<MyPacket*> locked_packets,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->stream = stream;
    this->locked_packets = locked_packets;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->transmition = transmition;
    this->creation_time = Sys::boostedTick();
}

PacketBundle::PacketBundle(
    Sys* sys,
                           BaseStream* stream,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->stream = stream;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->transmition = transmition;
    this->creation_time = Sys::boostedTick();
}

void PacketBundle::send_to_MA() {
    sys->memBus->send_from_NPU_to_MA(transmition, size, !needs_processing, send_back, this);
}

void PacketBundle::send_to_NPU() {
    sys->memBus->send_from_MA_to_NPU(transmition, size, !needs_processing, send_back, this);
}

void PacketBundle::call(EventType event, CallData* data) {
    if (needs_processing == true) {
        needs_processing = false;
        // this->delay[ns], size[bytes] local_mem_bw[bytes/s]
        this->delay = static_cast<uint64_t>(static_cast<double>(size) /
                                            sys->local_mem_bw * 1e9)  // write
                      + static_cast<uint64_t>(static_cast<double>(size) /
                                              sys->local_mem_bw * 1e9)  // read
                      + static_cast<uint64_t>(static_cast<double>(size) /
                                              sys->local_mem_bw * 1e9);  // read
        sys->try_register_event(this, EventType::CommProcessingFinished, data,
                                this->delay);
        return;
    }
    if (stream != nullptr && sys->valid_streams.find(stream) != sys->valid_streams.end()) {
        stream->call(EventType::General, data);
    }
    delete this;
}
