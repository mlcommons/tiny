#include "RawI2SWaveSink.hpp"

//#include "stm32h5xx_hal_gpio.h"
#include "stm32h573i_discovery_audio.h"
#include "tx_semaphore.h"

static INT active_buffer = -1;
static TX_SEMAPHORE buffer_semaphore;
//static DMA_HandleTypeDef handle_GPDMA1_Channel7;

//static SAI_HandleTypeDef hsai_BlockA2;

/**
  * @brief Tx Transfer completed callbacks.
  * @param  hsai : pointer to a SAI_HandleTypeDef structure that contains
  *                the configuration information for SAI module.
  * @retval None
  */
//void BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t instance)
//{
//  if(instance == 0)
//  {
//    active_buffer = 1;
//    tx_semaphore_ceiling_put(&buffer_semaphore, 1);
//  }
//}
//
///**
//  * @brief Tx Transfer Half completed callbacks
//  * @param  hsai : pointer to a SAI_HandleTypeDef structure that contains
//  *                the configuration information for SAI module.
//  * @retval None
//  */
//void BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t instance)
//{
//  if(instance == 0)
//  {
//    active_buffer = 0;
//    tx_semaphore_ceiling_put(&buffer_semaphore, 1);
//  }
//}

namespace Audio
{
  RawI2SWaveSink::RawI2SWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool)
        : WaveSink(runner, byte_pool)
  {
    if(buffer_semaphore.tx_semaphore_id != TX_SEMAPHORE_ID)
    {
      const char *name = "Raw I2S buffer playback semaphore";
      tx_semaphore_create(&buffer_semaphore, (char *)name, 0);
    }
  }

  PlayerState RawI2SWaveSink::GetState()
  {
    ULONG state;
    BSP_AUDIO_OUT_GetState(0, &state);
    return state == AUDIO_OUT_STATE_RESET
          ? RESET
          : (state == AUDIO_OUT_STATE_STOP ? STOPPED : UNKNOWN);
  }

  PlayerState RawI2SWaveSink::Initialize()
  {
    BSP_AUDIO_Init_t init;
    init.BitsPerSample = AUDIO_RESOLUTION_16B;
    init.ChannelsNbr = 2;
    init.Device = AUDIO_OUT_DEVICE_HEADPHONE;
    init.SampleRate = AUDIO_FREQUENCY_44K;
    init.Volume = 80;

    BSP_AUDIO_OUT_Init(0, &init);
    return GetState();
  }

  void RawI2SWaveSink::Configure(const WaveSource &source)
  {
    BSP_AUDIO_OUT_SetBitsPerSample(0, source.GetSampleSize());
    BSP_AUDIO_OUT_SetChannelsNbr(0, source.GetChannelCount());
    BSP_AUDIO_OUT_SetSampleRate(0, source.GetFrequency());
  }

  PlayerResult RawI2SWaveSink::Play(UCHAR *buffer, ULONG size)
  {
    return BSP_AUDIO_OUT_Play(0, buffer, size) == BSP_ERROR_NONE ? SUCCESS : ERROR;
  }

  PlayerResult RawI2SWaveSink::Stop()
  {
    return BSP_AUDIO_OUT_Stop(0) == BSP_ERROR_NONE ? SUCCESS : ERROR;
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
