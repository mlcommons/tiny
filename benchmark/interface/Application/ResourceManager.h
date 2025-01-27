#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H

#include "tx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void InitializeResourceManager(TX_BYTE_POOL *byte_pool);
void SetFxMedia(FX_MEDIA *media);

#ifdef __cplusplus
}
#endif

#endif //RESOURCEMANAGER_H
