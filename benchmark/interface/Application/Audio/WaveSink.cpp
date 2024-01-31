//
// Created by Steve Reckamp on 12/22/23.
//

#include "WaveSink.hpp"
#include "../Tasks/ITask.hpp"

#include "stm32h573i_discovery_audio.h"
#include "tx_semaphore.h"

#define PLAY_BUFFER_BYTES   8 * 1024

extern "C"
{
  static INT active_buffer = -1;
  static TX_SEMAPHORE buffer_semaphore;

  static void InitBufferSemaphore()
  {
    if(buffer_semaphore.tx_semaphore_id != TX_SEMAPHORE_ID)
    {
      const char *name = "Audio buffer playback semaphore";
      tx_semaphore_create(&buffer_semaphore, (char *)name, 0);
    }
  }

  /**
    * @brief Tx Transfer completed callbacks.
    * @param  hsai : pointer to a SAI_HandleTypeDef structure that contains
    *                the configuration information for SAI module.
    * @retval None
    */
  void BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t instance)
  {
    if(instance == 0)
    {
      active_buffer = 1;
      tx_semaphore_ceiling_put(&buffer_semaphore, 1);
    }
  }

  /**
    * @brief Tx Transfer Half completed callbacks
    * @param  hsai : pointer to a SAI_HandleTypeDef structure that contains
    *                the configuration information for SAI module.
    * @retval None
    */
  void BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t instance)
  {
    if(instance == 0)
    {
      active_buffer = 0;
      tx_semaphore_ceiling_put(&buffer_semaphore, 1);
    }
  }
}

namespace Audio
{
  class PlayWaveTask: public Tasks::ITask
  {
  public:
    PlayWaveTask(WaveSink &player, WaveSource &source):
        ITask(TX_FALSE), player(player), source(source)
        { }

    void Run()
    {
      result = player.AsyncPlay(source);
    }

    UCHAR GetResult()
    {
      Wait();
      return result;
    }
  private:
    WaveSink &player;
    WaveSource &source;
    UCHAR result;
  };

  WaveSink::WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool): runner(runner), size(PLAY_BUFFER_BYTES)
  {
    InitBufferSemaphore();

    tx_byte_allocate(&byte_pool, (void **)&play_buffer, size, TX_NO_WAIT);
  }

  UCHAR WaveSink::GetInfo(TX_QUEUE * const queue, const WaveSource &source)
  {
    std::string *msg = new std::string(source.GetName());
    msg->append("\n");
    tx_queue_send(queue, &msg, TX_WAIT_FOREVER);

    msg = new std::string(source.GetFormat() == 1 ? "non-PCM\n" : "PCM\n");
    tx_queue_send(queue, &msg, TX_WAIT_FOREVER);

    msg = new std::string(source.GetChannelCount() == 2 ? "stereo\n" : "mono\n");
    tx_queue_send(queue, &msg, TX_WAIT_FOREVER);

    char buf[50];
    snprintf(buf, sizeof(buf), "%dHz\n", source.GetFrequency());
    msg = new std::string(buf);
    tx_queue_send(queue, &msg, TX_WAIT_FOREVER);

    snprintf(buf, sizeof(buf), "%d-bit\n", source.GetSampleSize());
    msg = new std::string(buf);
    tx_queue_send(queue, &msg, TX_WAIT_FOREVER);

    return TX_TRUE;
  }

  UCHAR WaveSink::Play(WaveSource &source)
  {
    PlayWaveTask *task = new PlayWaveTask(*this, source);
    runner.Submit(task);
    UCHAR result = task->GetResult();
    delete task;
    return result;
  }

  UCHAR WaveSink::AsyncPlay(WaveSource &source)
  {
    ULONG state;
    BSP_AUDIO_OUT_GetState(0, &state);
    if(state == AUDIO_OUT_STATE_RESET)
    {
      BSP_AUDIO_Init_t init;
      init.BitsPerSample = AUDIO_RESOLUTION_16B;
      init.ChannelsNbr = 2;
      init.Device = AUDIO_OUT_DEVICE_HEADPHONE;
      init.SampleRate = AUDIO_FREQUENCY_44K;
      init.Volume = 80;

      BSP_AUDIO_OUT_Init(0, &init);
    }
    BSP_AUDIO_OUT_GetState(0, &state);
    if(state != AUDIO_OUT_STATE_STOP)
    {
      return TX_FALSE;
    }

    if(source.Open() == TX_TRUE)
    {
      BSP_AUDIO_OUT_SetBitsPerSample(0, source.GetSampleSize());
      BSP_AUDIO_OUT_SetChannelsNbr(0, source.GetChannelCount());
      BSP_AUDIO_OUT_SetSampleRate(0, source.GetFrequency());
      source.Seek(0);

      active_buffer = 0;
      ULONG next_bytes = source.ReadData(play_buffer, size);
      active_buffer = -1;

      UCHAR status = BSP_AUDIO_OUT_Play(0, (uint8_t *)play_buffer, size);

      while(status == BSP_ERROR_NONE && next_bytes > 0)
      {
        while(active_buffer == -1) tx_semaphore_get(&buffer_semaphore, 50);

        next_bytes = source.ReadData(&play_buffer[active_buffer * size/2], size / 2);
        active_buffer = -1;
      }
      BSP_AUDIO_OUT_Stop(0);
      source.Close();
    }

    return TX_TRUE;
  }
}
