#ifndef AUDIO_HEADPHONEWAVESINK_HPP
#define AUDIO_HEADPHONEWAVESINK_HPP

#include "WaveSink.hpp"

namespace Audio
{
  class HeadphoneWaveSink : public WaveSink
  {
  public:
    HeadphoneWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);
  protected:
    void Configure(const WaveSource &source);
    PlayerState GetState();
    PlayerState Initialize();
    PlayerResult Play(UCHAR *buffer, ULONG size);
    PlayerResult Stop();
  };
}

#endif // AUDIO_HEADPHONEWAVESINK_HPP
