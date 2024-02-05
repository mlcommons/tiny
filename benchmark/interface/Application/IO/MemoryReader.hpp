//
// Created by Steve Reckamp on 12/21/23.
//

#ifndef MEMORY_MEMORYREADER_HPP
#define MEMORY_MEMORYREADER_HPP

#include "IDataSource.hpp"

namespace IO
{
  class MemoryReader : public ::IDataSource
  {
  public:
    MemoryReader(ULONG address, ULONG size);

    const std::string &GetName() const { return name; }

    UCHAR Open() { return TX_TRUE; }

    UCHAR Close() { return TX_TRUE; }

    UCHAR Seek(ULONG position);

    ULONG ReadData(void *dest, ULONG length);

    ULONG GetPosition() const { return index; }

  private:
    std::string name;
    CHAR *const buffer;
    const ULONG size;
    ULONG index;
  };
}

#endif //MEMORY_MEMORYREADER_HPP
