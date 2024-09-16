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
  /**
   * Results of playing audio
   */
  typedef enum
  {
    SUCCESS,
    ERROR
  } PlayerResult;

  /**
   * State if the audio player
   */
  typedef enum
  {
    RESET,
    STOPPED,
    PLAYING,
    UNKNOWN
  } PlayerState;

  class PlayWaveTask;

  /**
   * Class to playback a wave file
   */
  class WaveSink
  {
  public:
    /**
     * Play a file
     * @param source The data to play
     * @return The result of the playback
     */
    PlayerResult Play(WaveSource &source);
    virtual ~WaveSink() { }
  protected:
    /**
     * Constructor
     * @param runner TaskRunner that executes the submitted Task::ITasks
     * @param byte_pool Memory to create the play buffer bytes
     */
    WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool);

    /**
     * Configure the playback device for the source
     * @param source The media to play
     * @return The results of the operation
     */
    virtual PlayerResult Configure(const WaveSource &source) = 0;

    /**
     * Read the state of the sink
     * @return Current state
     */
    virtual PlayerState GetState() = 0;

    /**
     * Do any one-time initialization required at startup
     * @return Current state
     */
    virtual PlayerState Initialize() = 0;

    /**
     * Play a buffer
     * @param buffer data to play
     * @param size size of the bugger to play
     * @return The results of the operation
     */
    virtual PlayerResult Play(UCHAR *buffer, ULONG size) = 0;

    /**
     * Stop the playback
     * @return The results of the operation
     */
    virtual PlayerResult Stop() = 0;
  private:
    friend void ::BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t);
    friend void ::BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t);
    static INT active_buffer;
    static TX_SEMAPHORE buffer_semaphore;
    Tasks::TaskRunner &runner;
    UCHAR *play_buffer;
    ULONG size;
    PlayerResult IndirectPlay(WaveSource &source);
    friend class PlayWaveTask;
  };
}

#endif //AUDIO_WAVESINK_HPP
