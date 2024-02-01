#include "HeadphoneWaveSink.hpp"

#include "stm32h573i_discovery_audio.h"

namespace Audio
{
  HeadphoneWaveSink::HeadphoneWaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool)
        : WaveSink(runner, byte_pool)
  {
  }

  PlayerState HeadphoneWaveSink::GetState()
  {
    ULONG state;
    BSP_AUDIO_OUT_GetState(0, &state);
    return state == AUDIO_OUT_STATE_RESET
          ? RESET
          : (state == AUDIO_OUT_STATE_STOP ? STOPPED : UNKNOWN);
  }

  PlayerState HeadphoneWaveSink::Initialize()
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

  void HeadphoneWaveSink::Configure(const WaveSource &source)
  {
    BSP_AUDIO_OUT_SetBitsPerSample(0, source.GetSampleSize());
    BSP_AUDIO_OUT_SetChannelsNbr(0, source.GetChannelCount());
    BSP_AUDIO_OUT_SetSampleRate(0, source.GetFrequency());
  }

  PlayerResult HeadphoneWaveSink::Play(UCHAR *buffer, ULONG size)
  {
    return BSP_AUDIO_OUT_Play(0, buffer, size) == BSP_ERROR_NONE ? SUCCESS : ERROR;
  }

  PlayerResult HeadphoneWaveSink::Stop()
  {
    return BSP_AUDIO_OUT_Stop(0) == BSP_ERROR_NONE ? SUCCESS : ERROR;
  }
}
