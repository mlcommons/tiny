#ifndef AUDIO_RAWI2SWAVESINK_HPP
#define AUDIO_RAWI2SWAVESINK_HPP

#include "WaveSink.hpp"

namespace Audio
{
  /**
   * Play audio to the I2S channel
   */
  class RawI2SWaveSink : public WaveSink
  {
  public:
    /**
     * Constructor
     * @param runner TaskRunner that executes the submitted Task::ITasks
     * @param byte_pool Memory to create the play buffer bytes
     */
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
