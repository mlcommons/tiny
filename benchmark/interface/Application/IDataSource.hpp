#ifndef IDATASOURCE_HPP
#define IDATASOURCE_HPP

#include "fx_api.h"
#include <string>

class IDataSource
{
public:
  IDataSource() {};
  virtual ~IDataSource() {};
  virtual const std::string &GetName() const = 0;
  virtual UCHAR Open() = 0;
  virtual UCHAR Close() = 0;
  virtual UCHAR Seek(ULONG position) = 0;
  virtual ULONG ReadData(void *dest, ULONG length) = 0;
  virtual ULONG GetPosition() const = 0;
};

#endif //IDATASOURCE_HPP
