#ifndef AUDIO_WAVESINK_HPP
#define AUDIO_WAVESINK_HPP

#include "tx_api.h"
#include "WaveSource.hpp"
#include "../Tasks/TaskRunner.hpp"

extern "C"
{
void BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t);
void BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t);
};

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
    virtual PlayerResult Configure(const WaveSource &source) = 0;
    virtual PlayerState GetState() = 0;
    virtual PlayerState Initialize() = 0;
    virtual PlayerResult Play(UCHAR *buffer, ULONG size) = 0;
    virtual PlayerResult Stop() = 0;
  private:
    friend void ::BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t);
    friend void ::BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t);
    static INT active_buffer;
    static TX_SEMAPHORE buffer_semaphore;
    Tasks::TaskRunner &runner;
    UCHAR *play_buffer;
    ULONG size;
    PlayerResult AsyncPlay(WaveSource &source);
    friend class PlayWaveTask;
  };
}

#endif //AUDIO_WAVESINK_HPP
