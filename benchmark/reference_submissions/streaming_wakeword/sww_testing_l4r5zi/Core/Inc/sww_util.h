/*
 * sww_util.h
 *
 *  Created on: Jan 16, 2025
 *      Author: jeremy
 */

#ifndef INC_SWW_UTIL_H_
#define INC_SWW_UTIL_H_

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>


#define EE_CMD_SIZE 80u
#define EE_CMD_DELIMITER " "
#define EE_CMD_TERMINATOR '%'

// struct for a log we can write to without the delay of printing to UART
#define LOG_BUFFER_SIZE 4096
typedef struct {
    char buffer[LOG_BUFFER_SIZE];
    size_t current_pos;
} LogBuffer;


void print_vals_int16(int16_t *buffer, uint32_t num_vals);
void print_bytes(uint8_t *buffer, uint32_t num_bytes);
void print_vals_float(float *buffer, uint32_t num_vals);
void log_printf(LogBuffer *log, const char *format, ...);

void process_command(char *full_command);
void ee_serial_callback(char c);

int aiInit(void);
void setup_i2s_buffers();


#endif /* INC_SWW_UTIL_H_ */
