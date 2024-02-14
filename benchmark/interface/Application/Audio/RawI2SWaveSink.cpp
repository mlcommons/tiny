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
    HAL_DMAEx_List_LinkQ(&handle_GPDMA2_Channel2, &RawSAIQueue);

    state = STOPPED;
    return GetState();
  }

  static LONG ConvertChannels(INT channelCount)
  {
    switch(channelCount)
    {
      case 1:
        return SAI_MONOMODE;
      case 2:
        return SAI_STEREOMODE;
      default:
        return -1;
    }
  }

  static LONG ConvertSampleSize(INT bitCount)
  {
    switch(bitCount)
    {
      case 16:
        return SAI_PROTOCOL_DATASIZE_16BIT;
      case 24:
        return SAI_PROTOCOL_DATASIZE_24BIT;
      case 32:
        return SAI_PROTOCOL_DATASIZE_32BIT;
      default:
        return -1;
    }
  }

  PlayerResult RawI2SWaveSink::Configure(const WaveSource &source)
  {
    hsai_BlockB1.Init.AudioFrequency = source.GetFrequency();
    hsai_BlockB1.Init.MonoStereoMode = ConvertChannels(source.GetChannelCount());
    if (HAL_SAI_InitProtocol(&hsai_BlockB1, SAI_I2S_STANDARD,
                             ConvertSampleSize(source.GetSampleSize()),
                             source.GetChannelCount()) != HAL_OK)
      return ERROR;
    return SUCCESS;
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
