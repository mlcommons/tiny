#ifndef AUDIO_WAVESOURCE_HPP
#define AUDIO_WAVESOURCE_HPP

#include "IDataSource.hpp"

namespace Audio
{
  /**
   * Read from a wave file
   */
  class WaveSource: ::IDataSource
  {
  public:
    /**
     * Allow direct casting from a data source to a WaveSource
     * @param source The data source
     */
    explicit WaveSource(IDataSource &source);
    ~WaveSource() { }
    /**
     * Get the name of the source
     * @return The name
     */
    const std::string &GetName() const { return source.GetName(); }

    /**
     * @return The file format (1 = PCM)
     */
    USHORT GetFormat() const { return format; }

    /**
     * @return The number of channels
     */
    USHORT GetChannelCount() const { return channels; }

    /**
     * @return The sample frequency
     */
    UINT GetFrequency() const { return frequency; }

    /**
     * @return The audio data rate (bytes/s)
     */
    UINT GetDataRate() const { return data_rate; }

    /**
     * @return The number of bits per second
     */
    USHORT GetSampleSize() const { return bits_per_sample; }

    /**
     * @return The Size of teh data
     */
    USHORT GetSize() const { return data_size; }

    /**
     * @return TX_TRUE if the header has been read and is the data is valid
     */
    UCHAR IsReady() const { return header_read & valid_wave; }

    /**
     * Open and parse the file
     * @return TX_TRUE if acceptable
     */
    UCHAR Open();

    /**
     * Close the file
     * @return TX_TRUE if the file is closed
     */
    UCHAR Close();

    /**
     * Seek to a particular location in the file
     * @param position The location in the file to move to
     * @return TX_TRUE if successful
     */
    UCHAR Seek(ULONG position);

    /**
     * Read the data
     * @param dest the buffer to receive the data
     * @param length the max number of bytes to read
     * @return The actual number of bytes read
     */
    ULONG ReadData(void *dest, ULONG length);

    /**
     * @return The current position in the data
     */
    ULONG GetPosition() const { return source.GetPosition() - data_offset; };

    /**
     * Read info from the file and send it to the queue
     * @param queue The queue for the results
     * @return TX_TRUE if successful
     */
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
  };
}

#endif //AUDIO_WAVESOURCE_HPP
