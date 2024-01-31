#ifndef AUDIO_WAVESOURCE_HPP
#define AUDIO_WAVESOURCE_HPP

#include "IDataSource.hpp"

namespace Audio
{
  class WaveSource: ::IDataSource
  {
  public:
    explicit WaveSource(IDataSource &source);
    ~WaveSource() { }
    const std::string &GetName() const { return source.GetName(); }
    USHORT GetFormat() const { return format; }
    USHORT GetChannelCount() const { return channels; }
    UINT GetFrequency() const { return frequency; }
    UINT GetDataRate() const { return data_rate; }
    USHORT GetSampleSize() const { return bits_per_sample; }
    USHORT GetSize() const { return data_size; }
    UCHAR IsReady() const { return header_read & valid_wave; }
    UCHAR Open();
    UCHAR Close();
    UCHAR Seek(ULONG position);
    ULONG ReadData(void *dest, ULONG length);
    ULONG GetPosition() const { return data_index; };
    UCHAR GetInfo(TX_QUEUE * const queue);
  private:
    IDataSource &source;
    UCHAR header_read;
    UCHAR is_opened;
    UCHAR valid_wave;
    USHORT format;
    USHORT channels;
    UINT frequency;
    UINT data_rate;
    USHORT block_alignment;
    USHORT bits_per_sample;
    UINT data_size;
    UINT data_offset;
    UINT data_index;
  };
}

#endif //AUDIO_WAVESOURCE_HPP
