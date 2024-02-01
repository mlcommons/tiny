#include "RawI2SWaveSink.hpp"

#include "gpdma.h"
#include "linked_list.h"
#include "sai.h"
#include "stm32h5xx_hal_sai.h"
#include "tx_semaphore.h"

extern DMA_QListTypeDef RawSAIQueue;

namespace Audio
{
  RawI2SWaveSink::RawI2SWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool)
        : WaveSink(runner, byte_pool), state(RESET)
  {
  }

  PlayerState RawI2SWaveSink::GetState()
  {
    return state;
  }

  PlayerState RawI2SWaveSink::Initialize()
  {
    __HAL_LINKDMA(&hsai_BlockB1, hdmatx, handle_GPDMA2_Channel2);
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
}
