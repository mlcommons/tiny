#ifndef CLI_INTERFACEMENU_H
#define CLI_INTERFACEMENU_H

#include "tx_api.h"
#include "usart.h"

#ifdef __cplusplus
extern "C" {
#endif

void CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart);
void CLI_Run();

#ifdef __cplusplus
}
#endif

#endif //CLI_INTERFACEMENU_H
