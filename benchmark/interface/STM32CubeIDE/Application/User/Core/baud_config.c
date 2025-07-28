#include <stdint.h>             // <-- Fixes uint32_t issues
#include "stm32h5xx_hal.h"      // <-- Fixes HAL_StatusTypeDef and Flash HAL
#include "baud_config.h"        // Your header (if it exists)
#include "stm32h5xx_hal_flash.h"

#define CONFIG_FLASH_ADDR   ((uint32_t)0x080FF800U) // Ensure 16-byte aligned and valid range
#define CONFIG_FLASH_SECTOR FLASH_SECTOR_127        // Choose based on actual layout
#define CONFIG_FLASH_BANK   FLASH_BANK_1

typedef struct __attribute__((aligned(16))) {
    uint32_t baudrate;
    uint32_t reserved[3];
} FlashBaudRate_t;

static const FlashBaudRate_t *flash_data = (FlashBaudRate_t *)CONFIG_FLASH_ADDR;

uint32_t LoadBaudRateFromFlashWithDefault(uint32_t fallback)
{
    uint32_t baud = flash_data->baudrate;

    if (baud == 0xFFFFFFFF || baud == 0 || baud > 2000000) {
        return fallback;
    }

    return baud;
}
uint32_t LoadBaudRateFromFlash(void)
{
    return LoadBaudRateFromFlashWithDefault(DEFAULT_BAUDRATE);
}

HAL_StatusTypeDef SaveBaudRateToFlash(uint32_t baud)
{
    HAL_StatusTypeDef status;
    FlashBaudRate_t data_to_write = {
        .baudrate = baud,
        .reserved = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
    };

    HAL_FLASH_Unlock();

    FLASH_EraseInitTypeDef erase = {
        .TypeErase    = FLASH_TYPEERASE_SECTORS,
        .Banks        = CONFIG_FLASH_BANK,
        .Sector       = CONFIG_FLASH_SECTOR,
        .NbSectors    = 1
    };

    uint32_t sector_error = 0;
    status = HAL_FLASHEx_Erase(&erase, &sector_error);
    if (status != HAL_OK) {
        HAL_FLASH_Lock();
        return status;
    }

    status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_QUADWORD,
                               CONFIG_FLASH_ADDR,
                               (uint32_t)&data_to_write);

    HAL_FLASH_Lock();

    // Optional: Confirm write
    if (status == HAL_OK && flash_data->baudrate != baud) {
        return HAL_ERROR;
    }

    return status;
}
