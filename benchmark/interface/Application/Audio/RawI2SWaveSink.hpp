#ifndef AUDIO_RAWI2SWAVESINK_HPP
#define AUDIO_RAWI2SWAVESINK_HPP

#include "WaveSink.hpp"

namespace Audio
{
  class RawI2SWaveSink : public WaveSink
  {
  public:
    RawI2SWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);
  protected:
    PlayerResult Configure(const WaveSource &source);
    PlayerState GetState();
    PlayerState Initialize();
    PlayerResult Play(UCHAR *buffer, ULONG size);
    PlayerResult Stop();
  private:
    PlayerState state;
  };
}

#endif // AUDIO_RAWI2SWAVESINK_HPP
