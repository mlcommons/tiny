//
// Created by Steve Reckamp on 12/21/23.
//

#include "WaveSource.hpp"

namespace Audio
{
  WaveSource::WaveSource(IDataSource &source):
    source(source),
    header_read(TX_FALSE),
    is_opened(TX_FALSE),
    valid_wave(TX_FALSE),
    format(0),
    channels(0),
    frequency(0),
    data_rate(0),
    block_alignment(0),
    bits_per_sample(0),
    data_size(0),
    data_offset(0),
    data_index(0)
  { }

  /**
   * Implemented based on: https://isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p05.html
   */
  UCHAR WaveSource::Open()
  {
    UCHAR result = TX_TRUE;

    if(header_read != TX_TRUE)
    {
      char buffer[8];
      UINT size;
      USHORT format;
      USHORT channels;
      UINT frequency;
      UINT data_rate;
      USHORT block_alignment;
      USHORT bits_per_sample;
      UINT data_size;
      UINT chunk_size;

      result = source.Open();
      if(result)
      {
        is_opened = TX_TRUE;
        result = source.Seek(0);
      }

      if(result)
        result = source.ReadData(buffer, 4) == 4 && strncmp("RIFF", buffer, 4) == 0 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&size, 4) == 4 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(buffer, 8) == 8 && strncmp("WAVEfmt ", buffer, 8) == 0 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&chunk_size, 4) == 4 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&format, 2) == 2 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&channels, 2) == 2 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&frequency, 4) == 4 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&data_rate, 4) == 4 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&block_alignment, 2) == 2 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&bits_per_sample, 2) == 2 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(buffer, 4) == 4 && strncmp("data", buffer, 4) == 0 ? TX_TRUE : TX_FALSE;

      if(result)
        result = source.ReadData(&data_size, 4) == 4 ? TX_TRUE : TX_FALSE;

      if(result)
      {
        this->format = format;
        this->channels = channels;
        this->frequency = frequency;
        this->data_rate = data_rate;
        this->block_alignment = block_alignment;
        this->bits_per_sample = bits_per_sample;
        this->data_size = data_size;
        this->data_offset = source.GetPosition();
        valid_wave = TX_TRUE;
      }
      else if(is_opened)
      {
        Close();
      }

      header_read = TX_TRUE;
    }
    return result;
  }

  UCHAR WaveSource::Close()
  {
    if(is_opened == TX_FALSE)
      return TX_FALSE;
    is_opened = TX_FALSE;
    return source.Close();
  }

  UCHAR WaveSource::Seek(ULONG position)
  {
    if(position < data_size)
    {
      data_index = position;
      return TX_TRUE;
    }
    return TX_FALSE;
  }

  ULONG WaveSource::ReadData(void *dest, ULONG length)
  {
    if(is_opened == TX_FALSE) return 0;

    length = data_size > data_index + length ? length : data_size - data_index;
    ULONG actual_bytes = source.ReadData(dest, length);
    data_index = source.GetPosition();
    return actual_bytes;
  }
}
