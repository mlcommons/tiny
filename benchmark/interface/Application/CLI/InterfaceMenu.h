#ifndef CLI_INTERFACEMENU_H
#define CLI_INTERFACEMENU_H

#include "tx_api.h"
#include "usart.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create the menu
 * @param byte_pool memory to store the response queue
 * @param huart The host UART
 */
void CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart);
/**
 * Run the menu
 */
void CLI_Run();

void Record_WW_Detection();
void Record_Dutycycle_Stop();
void Record_Dutycycle_Start();

#ifdef __cplusplus
}
#endif

#endif //CLI_INTERFACEMENU_H
