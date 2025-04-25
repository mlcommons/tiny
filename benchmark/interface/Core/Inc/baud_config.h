#ifndef BAUD_CONFIG_H
#define BAUD_CONFIG_H

#include "stm32h5xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_BAUDRATE 9600U

// In baud_config.h
uint32_t LoadBaudRateFromFlash(void);
uint32_t LoadBaudRateFromFlashWithDefault(uint32_t fallback);
HAL_StatusTypeDef SaveBaudRateToFlash(uint32_t baud);

#ifdef __cplusplus
}
#endif

#endif // BAUD_CONFIG_H
