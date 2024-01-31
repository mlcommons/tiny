#ifndef AUDIO_WAVESINK_HPP
#define AUDIO_WAVESINK_HPP

#include "tx_api.h"
#include "WaveSource.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace Audio
{
  class PlayWaveTask;
  class WaveSink
  {
  public:
    WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);
    UCHAR Configure(const WaveSource &source);
    UCHAR Play(WaveSource &source);
  private:
    Tasks::TaskRunner &runner;
    UCHAR *play_buffer;
    ULONG size;
    UCHAR AsyncPlay(WaveSource &source);
    friend class PlayWaveTask;
  };
}

#endif //AUDIO_WAVESINK_HPP
