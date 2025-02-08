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
#include "sww_model.h"

#define EE_CMD_SIZE 80u
#define EE_CMD_DELIMITER " "
#define EE_CMD_TERMINATOR '%'

// struct for a log we can write to without the delay of printing to UART
#define LOG_BUFFER_SIZE 4096
typedef struct {
    char buffer[LOG_BUFFER_SIZE];
    size_t current_pos;
} LogBuffer;



typedef enum {
	FileCapture, // capture a block of a pre-determined size, then stop
	Streaming    // continually capture blocks of "window_step" size, then
	             // extract features and run NN

} i2s_mode_t;


void print_vals_int16(const int16_t *buffer, uint32_t num_vals);
void print_bytes(const uint8_t *buffer, uint32_t num_bytes);
void print_vals_float(const float *buffer, uint32_t num_vals);
void log_printf(LogBuffer *log, const char *format, ...);

void process_command(char *full_command);
void ee_serial_callback(char c);

ai_error aiInit(void);
void setup_i2s_buffers();
void compute_lfbe_f32(const int16_t *pSrc, float32_t *pDst, float32_t *pTmp);


#endif /* INC_SWW_UTIL_H_ */
