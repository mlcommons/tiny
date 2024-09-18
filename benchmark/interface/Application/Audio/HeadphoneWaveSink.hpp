#ifndef AUDIO_HEADPHONEWAVESINK_HPP
#define AUDIO_HEADPHONEWAVESINK_HPP

#include "WaveSink.hpp"

namespace Audio
{
  /**
   * Play audio to the headphone
   */
  class HeadphoneWaveSink : public WaveSink
  {
  public:
    /**
     * Constructor
     * @param runner TaskRunner that executes the submitted Task::ITasks
     * @param byte_pool Memory to create the play buffer bytes
     */
    HeadphoneWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);
  protected:
    PlayerResult Configure(const WaveSource &source);
    PlayerState GetState();
    PlayerState Initialize();
    PlayerResult Play(UCHAR *buffer, ULONG size);
    PlayerResult Stop();
  };
}

#endif // AUDIO_HEADPHONEWAVESINK_HPP
