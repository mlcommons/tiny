#include "WaveSink.hpp"
#include "../Tasks/ITask.hpp"

#include "tx_semaphore.h"

#define PLAY_BUFFER_BYTES   8 * 1024

// DMA: Need to do  variable-sized DMA transfers
// DMA: extern DMA_NodeTypeDef NodeTx;
// DMA: extern DMA_NodeTypeDef NodeTx2;

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
    Audio::WaveSink::active_buffer = 1;
    tx_semaphore_ceiling_put(&Audio::WaveSink::buffer_semaphore, 1);
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
    Audio::WaveSink::active_buffer = 0;
    tx_semaphore_ceiling_put(&Audio::WaveSink::buffer_semaphore, 1);
  }
}

namespace Audio
{
  class PlayWaveTask: public Tasks::IIndirectTask<WaveSink>
  {
  public:
    PlayWaveTask(WaveSink &player, WaveSource &source):
        Tasks::IIndirectTask<WaveSink>(player, TX_FALSE), source(source)
        {
        }

    void Run()
    {
      result = actor.IndirectPlay(source);
    }

    PlayerResult GetResult()
    {
      Wait();
      return result;
    }
  private:
    WaveSource &source;
    PlayerResult result;
  };

  INT WaveSink::active_buffer = -1;
  TX_SEMAPHORE WaveSink::buffer_semaphore;
// DMA:  DMA_NodeTypeDef *WaveSink::nodes[] = {&NodeTx, &NodeTx2};

  WaveSink::WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool): runner(runner), size(PLAY_BUFFER_BYTES)
  {
    if(buffer_semaphore.tx_semaphore_id != TX_SEMAPHORE_ID)
    {
      const char *name = "SAI buffer playback semaphore";
      tx_semaphore_create(&buffer_semaphore, (char *)name, 0);
    }
    tx_byte_allocate(&byte_pool, (void **)&play_buffer, size, TX_NO_WAIT);
  }

  PlayerResult WaveSink::Play(WaveSource &source)
  {
    PlayWaveTask *task = new PlayWaveTask(*this, source);
    runner.Submit(task);
    PlayerResult result = task->GetResult();
    delete task;
    return result;
  }

  PlayerResult WaveSink::IndirectPlay(WaveSource &source)
  {
    PlayerState state = GetState();
    if(state == RESET)
    {
      state = Initialize();
    }
    if(state != STOPPED)
    {
      return ERROR;
    }

    PlayerResult result = ERROR;

    if(source.Open() == TX_TRUE)
    {
      result = Configure(source);
      if(result == SUCCESS)
      {
        source.Seek(0);

        ULONG next_bytes = source.ReadData(play_buffer, size);
        active_buffer = -1;

        result = Play((UCHAR *)play_buffer, size);
// DMA:        result = Play((UCHAR *)play_buffer, next_bytes);
// DMA:        DMA_NodeConfTypeDef node1c, node0c;
// DMA:        HAL_DMAEx_List_GetNodeConfig(&node0c, nodes[0]);
// DMA:        HAL_DMAEx_List_GetNodeConfig(&node1c, nodes[1]);
// DMA:        if(result == SUCCESS)
// DMA:        {
// DMA:          state = PLAYING;
// DMA:          printf ("Playing %s!\r\n", source.GetName().c_str());
// DMA:        while(status == SUCCESS && next_bytes > 0)
// DMA:          while(next_bytes > 0)

        while(status == SUCCESS && next_bytes > 0)
          {
            while(active_buffer == -1) tx_semaphore_get(&buffer_semaphore, 50);
            INT buffer_idx = active_buffer;
            active_buffer = -1;

// DMA:            HAL_DMAEx_List_GetNodeConfig(&node0c, nodes[0]);
// DMA:            HAL_DMAEx_List_GetNodeConfig(&node1c, nodes[1]);
            next_bytes = source.ReadData(&play_buffer[buffer_idx * size/2], size / 2);
// DMA:            printf ("Prepared %ld bytes\r\n", next_bytes);
          }
// DMA:          printf ("Stop\r\n");
          Stop();
// DMA:        }
      }
      source.Close();
    }

    return result;
  }
}
