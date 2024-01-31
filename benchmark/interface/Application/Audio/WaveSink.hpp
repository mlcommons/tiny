#ifndef AUDIO_WAVESINK_HPP
#define AUDIO_WAVESINK_HPP

#include "tx_api.h"
#include "WaveSource.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace Audio
{
  typedef enum
  {
    SUCCESS,
    ERROR
  } PlayerResult;

  typedef enum
  {
    RESET,
    STOPPED,
    PLAYING,
    UNKNOWN
  } PlayerState;

  class PlayWaveTask;
  class WaveSink
  {
  public:
    WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);
    virtual ~WaveSink() { }
    PlayerResult Play(WaveSource &source);
  protected:
    virtual void Configure(const WaveSource &source) = 0;
    virtual PlayerState GetState() = 0;
    virtual PlayerState Initialize() = 0;
    virtual PlayerResult Play(UCHAR *buffer, ULONG size) = 0;
    virtual PlayerResult Stop() = 0;
    virtual INT WaitForActiveBuffer() = 0;
    virtual void SetActiveBuffer(INT buffer_id) = 0;
  private:
    Tasks::TaskRunner &runner;
    UCHAR *play_buffer;
    ULONG size;
    PlayerResult AsyncPlay(WaveSource &source);
    friend class PlayWaveTask;
  };
}

#endif //AUDIO_WAVESINK_HPP
