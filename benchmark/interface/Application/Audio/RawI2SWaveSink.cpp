#include "RawI2SWaveSink.hpp"

#include "gpdma.h"
#include "linked_list.h"
#include "sai.h"
#include "stm32h5xx_hal_sai.h"
#include "tx_semaphore.h"

extern DMA_QListTypeDef RawSAIQueue;

static INT active_buffer = -1;
static TX_SEMAPHORE buffer_semaphore;

/**
  * @brief Tx Transfer completed callbacks.
  * @param  hsai : pointer to a SAI_HandleTypeDef structure that contains
  *                the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
  if(hsai == &hsai_BlockB1)
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
void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  if(hsai == &hsai_BlockB1)
  {
    active_buffer = 0;
    tx_semaphore_ceiling_put(&buffer_semaphore, 1);
  }
}

namespace Audio
{
  RawI2SWaveSink::RawI2SWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool)
        : WaveSink(runner, byte_pool), state(RESET)
  {
    if(buffer_semaphore.tx_semaphore_id != TX_SEMAPHORE_ID)
    {
      const char *name = "Raw I2S buffer playback semaphore";
      tx_semaphore_create(&buffer_semaphore, (char *)name, 0);
    }
  }

  PlayerState RawI2SWaveSink::GetState()
  {
    return state;
  }

  PlayerState RawI2SWaveSink::Initialize()
  {
    __HAL_LINKDMA(&hsai_BlockB1, hdmatx, handle_GPDMA1_Channel7);
    MX_RawSAIQueue_Config();
    HAL_DMAEx_List_SetCircularMode(&RawSAIQueue);

    state = STOPPED;
    return GetState();
  }

  void RawI2SWaveSink::Configure(const WaveSource &source)
  {
//    BSP_AUDIO_OUT_SetBitsPerSample(0, source.GetSampleSize());
//    BSP_AUDIO_OUT_SetChannelsNbr(0, source.GetChannelCount());
//    BSP_AUDIO_OUT_SetSampleRate(0, source.GetFrequency());
  }

  PlayerResult RawI2SWaveSink::Play(UCHAR *buffer, ULONG size)
  {
    if(state == STOPPED)
    {
      HAL_SAI_Transmit_DMA(&hsai_BlockB1, buffer, size);
      state = PLAYING;
      return SUCCESS;
    }
    return ERROR;
  }

  PlayerResult RawI2SWaveSink::Stop()
  {
    if(state == PLAYING)
    {
      HAL_SAI_DMAStop(&hsai_BlockB1);
      state = STOPPED;
      return SUCCESS;
    }
    return ERROR;
  }

  INT RawI2SWaveSink::WaitForActiveBuffer()
  {
    while(active_buffer == -1) tx_semaphore_get(&buffer_semaphore, 50);
    return active_buffer;
  }

  void RawI2SWaveSink::SetActiveBuffer(INT buffer_id)
  {
    active_buffer = buffer_id;
  }
}
