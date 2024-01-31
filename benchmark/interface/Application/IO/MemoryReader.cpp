//
// Created by Steve Reckamp on 12/21/23.
//

#include "MemoryReader.hpp"

namespace IO
{
  MemoryReader::MemoryReader(ULONG address, ULONG size) : buffer(reinterpret_cast<CHAR *>(address)), size(size),
                                                          index(0)
  {
    char buf[13];
    snprintf(buf, sizeof(buf), "@0x%08lX", address);
    name.append(buf);
  }

  UCHAR MemoryReader::Seek(ULONG position)
  {
    if (position < size)
    {
      index = position;
      return TX_TRUE;
    }
    return TX_FALSE;
  };

  ULONG MemoryReader::ReadData(void *dest, ULONG length)
  {
    length = size > index + length ? length : size - index;
    memcpy(dest, &buffer[index], length);
    index += length;
    return length;
  }
}
