/*
 * submitter_implemented.c
 *
 *  Created on: Sep 3, 2025
 *      Author: owen
 */

#include "sww_ref_util_submitter.h"

// private variables from main.c
UART_HandleTypeDef hlpuart1;
UART_HandleTypeDef huart3;

SAI_HandleTypeDef hsai_BlockA1;
DMA_HandleTypeDef hdma_sai1_a;

TIM_HandleTypeDef htim16;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

extern i2s_state_t g_i2s_state;

// variables from ST middlewares, only for use on ST hardware
/* Global handle to reference the instantiated C-model */
static ai_handle sww_model = AI_HANDLE_NULL;

/* Global c-array to handle the activations buffer */
AI_ALIGNED(32)
static ai_i8 activations[AI_SWW_MODEL_DATA_ACTIVATIONS_SIZE];

/* Array to store the data of the input tensor */
AI_ALIGNED(32)
static ai_i8 in_data[AI_SWW_MODEL_IN_1_SIZE];
/* or static ai_i8 in_data[AI_SWW_MODEL_DATA_IN_1_SIZE_BYTES]; */

/* c-array to store the data of the output tensor */
AI_ALIGNED(32)
static ai_i8 out_data[AI_SWW_MODEL_OUT_1_SIZE];
/* static ai_i8 out_data[AI_SWW_MODEL_DATA_OUT_1_SIZE_BYTES]; */

/* Array of pointer to manage the model's input/output tensors */
static ai_buffer *ai_input;
static ai_buffer *ai_output;

// from sw_ref_util, for platform-specific I2S functionality
extern uint32_t g_i2s_chunk_size_bytes;
extern int8_t *g_model_input;
extern uint32_t g_first_frame;
extern LogBuffer g_log;
extern int16_t *g_wav_block_buff;
extern int16_t *g_i2s_buffer0;
extern int16_t *g_i2s_buffer1;
extern int16_t *g_i2s_current_buff;
extern int g_i2s_buff_sel;
extern uint32_t g_gp_buff_bytes;
extern int8_t *g_act_buff;
extern uint32_t g_i2s_status;
extern uint32_t g_act_idx;

// static function prototypes
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_LPUART1_UART_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_SAI1_Init(void);
static void MX_TIM16_Init(void);

PUTCHAR_PROTOTYPE
{
  HAL_UART_Transmit(&hlpuart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
  return ch;
}

/// Core API function implementations
void th_delay_us(int delay_len_us)
{
	// there may be a better way to implement this
	// this will not give an accurate 1us delay, but
	// for longer delays it should be accurate to within 1us.
	int delay_start = __HAL_TIM_GET_COUNTER(&htim16);
	while (__HAL_TIM_GET_COUNTER(&htim16) < delay_start + 1);
}

void th_hardware_init(void)
{
	/* MCU Configuration------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the
       Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_DMA_Init();
	MX_LPUART1_UART_Init();
	MX_USART3_UART_Init();
	MX_USB_OTG_FS_PCD_Init();
	MX_SAI1_Init();
	MX_TIM16_Init();
}

// implementation of th_timer16_start
void th_timer16_start(void) { HAL_TIM_Base_Start(&htim16); }

// implementation of th_timer16_get
//uint16_t th_timer16_get(void) { return __HAL_TIM_GET_COUNTER(&htim16); }

// implementation of th_dma_receive
uint32_t th_dma_receive(uint8_t *i2s_buffer, uint16_t size)
{
    return HAL_SAI_Receive_DMA(&hsai_BlockA1, i2s_buffer, size);
}

// implementation of th_dma_stop
uint32_t th_dma_stop(void)
{
    return HAL_SAI_DMAStop(&hsai_BlockA1);
}

// implementation of th_dma_stae
uint8_t th_dma_state(void)
{
    return hsai_BlockA1.State;
}

// implementation of th_uart_receive
uint32_t th_uart_receive(uint8_t *data, uint16_t size, uint32_t timeout)
{
    return HAL_UART_Receive(&hlpuart1, data, size, timeout);
}

/*
 * Bootstrap inference framework
 */
ai_error th_ai_init(void)
{
	ai_error err;

	/* Create and initialize the c-model */
	const ai_handle acts[] = { activations };
	err = ai_sww_model_create_and_init(&sww_model, acts, NULL);

	if (err.type != AI_ERROR_NONE)
	{
		;
	};

	/* Reteive pointers to the model's input/output tensors */
	ai_input = ai_sww_model_inputs_get(sww_model, NULL);
	ai_output = ai_sww_model_outputs_get(sww_model, NULL);

	return err;
}

/*
 * Run inference
 */
ai_error th_ai_run(const void *in_data, void *out_data)
{
	ai_i32 n_batch;
	ai_error err;

	/* 1 - Update IO handlers with the data payload */
	ai_input[0].data = AI_HANDLE_PTR(in_data);
	ai_output[0].data = AI_HANDLE_PTR(out_data);

	/* 2 - Perform the inference */
	n_batch = ai_sww_model_run(sww_model, &ai_input[0], &ai_output[0]);
	if (n_batch != 1)
		err = ai_sww_model_get_error(sww_model);

	return err;
}

// implementation of th_run_model_on_test_data
void th_run_model_on_test_data(char *cmd_args[])
{
//	acquire_and_process_data(in_data);
	const int8_t *input_source=NULL;
	uint16_t timer_start, timer_stop, timer_diff;

	printf("In run_model. about to run model\r\n");
	if (strcmp(cmd_args[1], "class0") == 0)
		input_source = test_input_class0;
	else if (strcmp(cmd_args[1], "class1") == 0)
		input_source = test_input_class1;
	else if (strcmp(cmd_args[1], "class2") == 0)
		input_source = test_input_class2;
	else
	{
		printf("Unknown input tensor name, defaulting to test_input_class0\r\n");
		input_source = test_input_class0;
	}
	for (int i = 0 ; i < AI_SWW_MODEL_IN_1_SIZE ; i++)
		in_data[i] = (ai_i8)input_source[i];
	ee_set_processing_pin_high();
	timer_start = __HAL_TIM_GET_COUNTER(&htim16);
	/*  Call inference engine */
	th_ai_run(in_data, out_data);
	timer_stop = __HAL_TIM_GET_COUNTER(&htim16);
	ee_set_processing_pin_low();
	timer_diff = timer_stop-timer_start;
	printf("TIM16: th_ai_run took (%u : %u) = %u TIM16 cycles\r\n", timer_start,
			timer_stop, timer_diff);

	printf("Output = [");
	for (int i = 0 ; i < AI_SWW_MODEL_OUT_1_SIZE ; i++)
		printf("%02d, ", out_data[i]);
	printf("]\r\n");
}

// implementation of th_infer_static_wav
void th_infer_static_wav(char *cmd_args[])
{
	// feature_buff is used internally as a 2nd internal scratch space,
	// in the FFT domain, so it needs to be winlen_samples long, even though
	// ultimately it will only hold NUM_MEL_FILTERS values.  This can probably
	// be improved with a refactored th_compute_lfbe_f32().
	static float32_t feature_buff[SWW_WINLEN_SAMPLES];
	static float32_t dsp_buff[SWW_WINLEN_SAMPLES];
	int num_steps;  // jhdbg
	int offset;
	uint32_t wav_len=0;
	const int16_t *wav_ptr=NULL;

	offset = atoi(cmd_args[1]);
	wav_ptr = test_wav_long + offset;
	wav_len = test_wav_long_len-offset;
	printf("Infering on static wav with offset = %d\r\n", offset);

	num_steps = (wav_len - (SWW_WINLEN_SAMPLES - SWW_WINSTRIDE_SAMPLES))
                / SWW_WINSTRIDE_SAMPLES;

	// extract the input scale factor from the (file-global) ai_input
	float32_t input_scale_factor
        = *(ai_input[0].meta_info->intq_info->info->scale);

	// initialize model input buffer to 0s.
	for (int i = 0 ; i < SWW_MODEL_INPUT_SIZE ; i++)
		g_model_input[i] = 0;

	for (int idx_step = 0 ; idx_step < num_steps ; idx_step++)
	{

		th_compute_lfbe_f32(wav_ptr+(idx_step*SWW_WINSTRIDE_SAMPLES),
            feature_buff, dsp_buff);

		// shift current features in g_model_input[] and add new ones.
		for (int i = 0 ; i < SWW_MODEL_INPUT_SIZE - NUM_MEL_FILTERS ; i++)
			g_model_input[i] = g_model_input[i+NUM_MEL_FILTERS];

		for (int i = 0 ; i < NUM_MEL_FILTERS ; i++)
			g_model_input[i+SWW_MODEL_INPUT_SIZE-NUM_MEL_FILTERS]
                = (int8_t)(feature_buff[i]/input_scale_factor-128);

		for (int i = 0 ; i < AI_SWW_MODEL_IN_1_SIZE ; i++)
			in_data[i] = (ai_i8)g_model_input[i];

		// print out the newest vector of features as int8
		printf("(");
		ee_print_vals_int8(g_model_input+SWW_MODEL_INPUT_SIZE-NUM_MEL_FILTERS,
            NUM_MEL_FILTERS);
		printf(", ");

		/*  Call inference engine */
		th_ai_run(in_data, out_data);

		if (out_data[0] > DETECT_THRESHOLD || g_first_frame)
		{
			printf("[%d]: Detection (%d).  g_first_frame=%lu\r\n", idx_step,
                out_data[0], g_first_frame);
			ee_log_printf(&g_log, "[%d]: Detection (%d).  g_first_frame=%lu\r\n",
                idx_step, out_data[0], g_first_frame);
			g_first_frame = 0;
		}
		else if( out_data[0] > 100)
			printf("[%d]: Near miss (%d). \r\n", idx_step, out_data[0]);

		printf("%d), \r\n", out_data[0]);
	}
}

// implementation of th_extract_features_on_chunk
void th_extract_features_on_chunk(char *cmd_args[])
{
	// feature_buff is used internally as a 2nd internal scratch space,
	// in the FFT domain, so it needs to be winlen_samples long, even though
	// ultimately it will only hold NUM_MEL_FILTERS values.  This can probably
	// be improved with a refactored th_compute_lfbe_f32().
	static float32_t feature_buff[SWW_WINLEN_SAMPLES];
	static float32_t dsp_buff[SWW_WINLEN_SAMPLES];
	static int num_calls=0;

	// extract the input scale factor from the (file-global) ai_input
	float32_t input_scale_factor
        = *(ai_input[0].meta_info->intq_info->info->scale);


	if (num_calls == 0)
		for(int i=0;i<SWW_MODEL_INPUT_SIZE;i++)
			g_model_input[i] = 0;

	// wav samples should be in g_i2s_buffer0

    // g_wav_block_buff[SWW_WINSTRIDE_SAMPLES:<end>]  are old samples to be
    // shifted to the beginning of the clip. After this block,
    // g_wav_block_buff[0:(winlen-winstride)] is populated
    for (int i = SWW_WINSTRIDE_SAMPLES ; i < SWW_WINLEN_SAMPLES ; i++)
    	g_wav_block_buff[i-SWW_WINSTRIDE_SAMPLES] = g_wav_block_buff[i];

    // Now fill in g_wav_block_buff[(winlen-winstride):] with winstride new samples
	// no 2* here because UART transmits mono, unlike I2S buffer, which is stereo
	for (int i = SWW_WINLEN_SAMPLES - SWW_WINSTRIDE_SAMPLES ;
            i < SWW_WINLEN_SAMPLES ; i++)
		g_wav_block_buff[i]
            = g_i2s_buffer0[i-(SWW_WINLEN_SAMPLES-SWW_WINSTRIDE_SAMPLES)];

	th_compute_lfbe_f32(g_wav_block_buff, feature_buff, dsp_buff);

	// shift current features in g_model_input[] and add new ones.
	for (int i = 0 ; i < SWW_MODEL_INPUT_SIZE - NUM_MEL_FILTERS ; i++)
		g_model_input[i] = g_model_input[i + NUM_MEL_FILTERS];

	for (int i = 0 ; i < NUM_MEL_FILTERS ; i++)
		g_model_input[i + SWW_MODEL_INPUT_SIZE - NUM_MEL_FILTERS]
            = (int8_t)(feature_buff[i] / input_scale_factor - 128);

	for (int i = 0 ; i < AI_SWW_MODEL_IN_1_SIZE ; i++)
		in_data[i] = (ai_i8)g_model_input[i];

	/*  Call inference engine */
	th_ai_run(in_data, out_data);

    num_calls++;

	printf("m-features-[");
	for (int i = 0 ; i < NUM_MEL_FILTERS ; i++)
	{
		printf("%+3d", (int8_t)(feature_buff[i]/input_scale_factor-128));
		if (i < NUM_MEL_FILTERS -1)
			printf(", ");
	}
	printf("]\r\n");

	printf("m-activations-[%+3d, %+3d, %+3d]\r\n", out_data[0], out_data[1],
        out_data[2]);
}

// implementation of th_run_extraction
// internally-implemented for performance reasons (timer)
void th_run_extraction(char *cmd_args[])
{
	// Feature extraction work
	float32_t test_out[1024] = {0.0};
	float32_t dsp_buff[1024] = {0.0};
	// this will only operate on the first block_size (1024) elements of the
    // input wav

	uint32_t timer_start, timer_stop;
	char *endptr;
	uint32_t offset;

    // Optional offset arg.  "extract 1024"
    // if cmd_arg[1] is present, convert to long
    if (cmd_args[1] != NULL && *cmd_args[1] != '\0')
		offset = strtol(cmd_args[1], &endptr, 10);
	else
		offset = 0;
	timer_start = __HAL_TIM_GET_COUNTER(&htim16);
	th_compute_lfbe_f32(test_wav_marvin+offset, test_out, dsp_buff);
	timer_stop = __HAL_TIM_GET_COUNTER(&htim16);

	printf("TIM16: th_compute_lfbe_f32 took (%lu : %lu) = %lu TIM16 cycles\r\n",
        timer_start, timer_stop, timer_stop-timer_start);
	printf("\r\n{\r\n");
	printf("\"Input\": ");
	ee_print_vals_int16(test_wav_marvin+offset, 1024);
	printf(",\r\n \"Output\": ");
	ee_print_vals_float(test_out, 40);
	printf("}\r\n");
}

// implementation of th_process_chunk_and_cont_streaming
void th_process_chunk_and_cont_streaming(void *hsai)
{

	// feature_buff is used internally as a 2nd internal scratch space,
	// in the FFT domain, so it needs to be winlen_samples long, even though
	// ultimately it will only hold NUM_MEL_FILTERS values.  This can probably
	// be improved with a refactored th_compute_lfbe_f32().
	static float32_t feature_buff[SWW_WINLEN_SAMPLES];
	static float32_t dsp_buff[SWW_WINLEN_SAMPLES];
	static int num_calls = 0;  // jhdbg

    // start of processing, used for duty cycle measurement
	ee_set_processing_pin_high();

	// extract the input scale factor from the (file-global) ai_input
	float32_t input_scale_factor
        = *(ai_input[0].meta_info->intq_info->info->scale);

	// idle_buffer is the one that will be idle after we switch
	int16_t *idle_buffer = g_i2s_buff_sel ? g_i2s_buffer1 : g_i2s_buffer0;
	g_i2s_buff_sel = g_i2s_buff_sel ^ 1; // toggle between 0/1=>g_i2s_buffer0/1
    g_i2s_current_buff = g_i2s_buff_sel ? g_i2s_buffer1 : g_i2s_buffer0;

	g_i2s_status = th_dma_receive((uint8_t *)g_i2s_current_buff,
                    g_i2s_chunk_size_bytes/2);

    // g_wav_block_buff[SWW_WINSTRIDE_SAMPLES:<end>]  are old samples to be
    // shifted to the beginning of the clip. After this block,
    // g_wav_block_buff[0:(winlen-winstride)] is populated
    for (int i = SWW_WINSTRIDE_SAMPLES ; i < SWW_WINLEN_SAMPLES ; i++)
    	g_wav_block_buff[i - SWW_WINSTRIDE_SAMPLES] = g_wav_block_buff[i];

    // Now fill in g_wav_block_buff[(winlen-winstride):] with winstride new samples
	// 2* is because the I2S buffer is in stereo
	for (int i = SWW_WINLEN_SAMPLES - SWW_WINSTRIDE_SAMPLES
            ; i < SWW_WINLEN_SAMPLES ; i++)
		g_wav_block_buff[i]
            = idle_buffer[2*(i-(SWW_WINLEN_SAMPLES-SWW_WINSTRIDE_SAMPLES))];

	th_compute_lfbe_f32(g_wav_block_buff, feature_buff, dsp_buff);

	// shift current features in g_model_input[] and add new ones.
	for (int i = 0 ; i < SWW_MODEL_INPUT_SIZE - NUM_MEL_FILTERS ; i++)
		g_model_input[i] = g_model_input[i+NUM_MEL_FILTERS];

	for (int i=0 ; i < NUM_MEL_FILTERS ; i++)
		g_model_input[i+SWW_MODEL_INPUT_SIZE-NUM_MEL_FILTERS]
            = (int8_t)(feature_buff[i]/input_scale_factor-128);

	for (int i=0 ; i < AI_SWW_MODEL_IN_1_SIZE ; i++)
		in_data[i] = (ai_i8)g_model_input[i];

	/*  Call inference engine */
	th_ai_run(in_data, out_data);

	if (out_data[0] > DETECT_THRESHOLD || g_first_frame)
	{
 	    TH_GPIO_WRITE(WW_DETECTED_GPIO_Port, WW_DETECTED_Pin, GPIO_PIN_RESET);
	    th_delay_us(1);
	    TH_GPIO_WRITE(WW_DETECTED_GPIO_Port, WW_DETECTED_Pin, GPIO_PIN_SET);
	    g_first_frame = 0;
	}

    if (g_act_idx < (g_gp_buff_bytes / sizeof(g_act_buff[0])))
    	g_act_buff[g_act_idx++] = out_data[0];

    num_calls++;
    ee_set_processing_pin_low();  // end of processing
                                  // used for duty cycle measurement
}

// implementation of th_compute_lfbe_f32
void th_compute_lfbe_f32(const int16_t *pSrc, float32_t *pDst, float32_t *pTmp)
{
	const uint32_t block_length=SWW_WINLEN_SAMPLES;
	const float32_t inv_block_length=1.0/SWW_WINLEN_SAMPLES;
	const uint32_t spec_len = SWW_WINLEN_SAMPLES/2+1;
	const float32_t preemphasis_coef = 0.96875; // 1.0 - 2.0 ** -5;
	const float32_t power_offset = 52.0;
	const uint32_t num_filters = 40;
	int i; // for looping
	// to maintain continuity in pre-emphasis over segment boundaries
	static float32_t last_value = 0.0;
	arm_status op_result = ARM_MATH_SUCCESS;

	// convert int16_t pSrc to float32_t.  range [-32768:32767] => [-1.0,1.0)
	// WINLEN - WINSTRIDE of these have already been converted once, so a
    // little speedup
	// could probably be gained by factoring this out
    // into process_chunk_and_continue_streaming
	for (i = 0 ; i < block_length ; i++)
		pDst[i] = ((float32_t)pSrc[i]) / 32768.0;

	// Apply pre-emphasis:  zero-pad input by 1
    // then x' = x[1:]-pe_coeff*x[:-1], so len(x')==len(x)
	// Start by scaling w/ coeff; pTmp = preemphasis_coef * input
	arm_scale_f32(pDst, preemphasis_coef, pTmp, block_length);
	// calculate pDst[0] separately since it uses a value from the last segment
	pDst[0] = pDst[0] - last_value * preemphasis_coef;

	// in the next frame pDst[SWW_WINSTRIDE_SAMPLES-1] will be 1 sample older
    // than the 1st sample, so it will be used in the pre-emphasis for pDst[0]
	last_value = pDst[SWW_WINSTRIDE_SAMPLES - 1];

	// use pDst as a 2nd temp buffer pDst[1:] - pTmp => pDst[1:]
	arm_sub_f32 (pDst+1, pTmp, pDst+1, block_length-1);

	// apply hamming window to pDst and put results in pTmp.
	arm_mult_f32(pDst, hamm_win_1024, pTmp, block_length);


	/* RFFT based implementation */
	arm_rfft_fast_instance_f32 rfft_s;
	op_result = arm_rfft_fast_init_f32(&rfft_s, block_length);
	if (op_result != ARM_MATH_SUCCESS)
		printf("Error %d in arm_rfft_fast_init_f32", op_result);
	arm_rfft_fast_f32(&rfft_s,pTmp,pDst,0); // use config rfft_s
                                            // FFT(pTmp) => pDst, ifft=0

	// Now we need to take the magnitude of the spectrum.
    // For block_length=1024, it will be 513 elements
	// we'll use pTmp as an array of block_length/2+1 real values.
	// the N/2th element is real and stuck in pDst[1] (where fft[0].imag=0
    // should be), move that to pTmp[block_length/2]
	pTmp[block_length/2] = pDst[1]; // real value corresponding to fsamp/2
	pDst[1] = 0; // so now pDst[0,1] = real,imag elements at f=0
                 // (always real, so imag=0)
	arm_cmplx_mag_f32(pDst,pTmp,block_length/2); // mag(pDst) => pTmp
                                                 // pTmp[512] already set.

	//  powspec = (1 / data_config['window_size_samples']) * tf.square(magspec)
	arm_mult_f32(pTmp, pTmp,pDst, spec_len); // pDst[0:513] = pTmp[0:513]^2
	arm_scale_f32(pDst, inv_block_length, pTmp, spec_len);


	// The original lin2mel matrix is spec_len x num_filters, where each column
    // holds one mel filter, lin2mel_packed_<X>x<Y> has all the non-zero
    // elements packed together in one 1D array _filter_starts are the locations
    // in each *original* column where the non-zero elements start
	// _filter_lens is how many non-zero elements are in each original column
	// So the i_th filter start in lin2mel_packed at sum(_filter_lens[:i])
	// And the corresponding spectrum segment starts at
    // linear_spectrum[_filter_starts[i]]
	int lin2mel_coeff_idx = 0;
	/* Apply MEL filters; linear spectrum is now in pTmp[0:spec_len], put mel
       spectrum in pDst[0:num_filters] */
	for (i = 0 ; i < num_filters ; i++)
	{
		arm_dot_prod_f32 (pTmp+lin2mel_513x40_filter_starts[i],
				lin2mel_packed_513x40+lin2mel_coeff_idx,
				lin2mel_513x40_filter_lens[i],
				pDst+i);

		lin2mel_coeff_idx += lin2mel_513x40_filter_lens[i];
	}

	// powspec_max = tf.reduce_max(input_tensor=powspec)
	// powspec = tf.clip_by_value(powspec, 1e-30, powspec_max)
    // # prevent -infinity on log
	for (i = 0 ; i < num_filters ; i++)
		pDst[i] = (pDst[i] > 1e-30) ? pDst[i] : 1e-30;

	for (i = 0 ; i < num_filters ; i++)
		pDst[i] = 10*log10(pDst[i]);

	//log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
	arm_offset_f32 (pDst, power_offset, pDst, num_filters);
	arm_scale_f32(pDst, (1.0/64.0), pTmp, num_filters);

	//log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)
	for(i = 0 ; i < num_filters ; i++)
		pDst[i] = (pTmp[i] < 0.0) ? 0.0 : ((pTmp[i] > 1.0) ? 1.0 : pTmp[i]);
}

/// Private functions
// private functions, formerly from main.c, mainly STM Cube auto-generated stuff
/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /** Configure the main internal regulator output voltage
     */
    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST)
            != HAL_OK)
    {
        Error_Handler();
    }

    /** Initializes the RCC Oscillators according to the specified parameters
     * in the RCC_OscInitTypeDef structure.
     */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48
                                        | RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 2;
    RCC_OscInitStruct.PLL.PLLN = 30;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
    RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    /** Initializes the CPU, AHB and APB buses clocks
     */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{
    /* USER CODE BEGIN LPUART1_Init 0 */

    /* USER CODE END LPUART1_Init 0 */

    /* USER CODE BEGIN LPUART1_Init 1 */

    /* USER CODE END LPUART1_Init 1 */
    hlpuart1.Instance = LPUART1;
    hlpuart1.Init.BaudRate = 115200;
    hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
    hlpuart1.Init.StopBits = UART_STOPBITS_1;
    hlpuart1.Init.Parity = UART_PARITY_NONE;
    hlpuart1.Init.Mode = UART_MODE_TX_RX;
    hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;
    if (HAL_UART_Init(&hlpuart1) != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8)
            != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8)
            != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN LPUART1_Init 2 */

    /* USER CODE END LPUART1_Init 2 */
}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{
    /* USER CODE BEGIN USART3_Init 0 */

    /* USER CODE END USART3_Init 0 */

    /* USER CODE BEGIN USART3_Init 1 */

    /* USER CODE END USART3_Init 1 */
    huart3.Instance = USART3;
    huart3.Init.BaudRate = 115200;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;
    huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart3) != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8)
        != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8)
            != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN USART3_Init 2 */

    /* USER CODE END USART3_Init 2 */
}

/**
  * @brief SAI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SAI1_Init(void)
{
    /* USER CODE BEGIN SAI1_Init 0 */

    /* USER CODE END SAI1_Init 0 */

    /* USER CODE BEGIN SAI1_Init 1 */

    /* USER CODE END SAI1_Init 1 */
    hsai_BlockA1.Instance = SAI1_Block_A;
    hsai_BlockA1.Init.AudioMode = SAI_MODESLAVE_RX;
    hsai_BlockA1.Init.Synchro = SAI_ASYNCHRONOUS;
    hsai_BlockA1.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
    hsai_BlockA1.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
    hsai_BlockA1.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
    hsai_BlockA1.Init.MonoStereoMode = SAI_STEREOMODE;
    hsai_BlockA1.Init.CompandingMode = SAI_NOCOMPANDING;
    hsai_BlockA1.Init.TriState = SAI_OUTPUT_NOTRELEASED;
    if (HAL_SAI_InitProtocol(&hsai_BlockA1, SAI_I2S_STANDARD,
                            SAI_PROTOCOL_DATASIZE_16BIT, 2) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN SAI1_Init 2 */

    /* USER CODE END SAI1_Init 2 */
}

/**
  * @brief TIM16 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM16_Init(void)
{
    /* USER CODE BEGIN TIM16_Init 0 */

    /* USER CODE END TIM16_Init 0 */

    /* USER CODE BEGIN TIM16_Init 1 */

    /* USER CODE END TIM16_Init 1 */
    htim16.Instance = TIM16;
    htim16.Init.Prescaler = 120-1;
    htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim16.Init.Period = 65535;
    htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim16.Init.RepetitionCounter = 0;
    htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN TIM16_Init 2 */
    HAL_TIM_Base_MspInit(&htim16);
    /* USER CODE END TIM16_Init 2 */
}

/**
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_FS_PCD_Init(void)
{
    /* USER CODE BEGIN USB_OTG_FS_Init 0 */

    /* USER CODE END USB_OTG_FS_Init 0 */

    /* USER CODE BEGIN USB_OTG_FS_Init 1 */

    /* USER CODE END USB_OTG_FS_Init 1 */
    hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
    hpcd_USB_OTG_FS.Init.dev_endpoints = 6;
    hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
    hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
    hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
    hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
    hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
    hpcd_USB_OTG_FS.Init.battery_charging_enable = ENABLE;
    hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
    hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
    if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN USB_OTG_FS_Init 2 */

    /* USER CODE END USB_OTG_FS_Init 2 */
}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{
    /* DMA controller clock enable */
    __HAL_RCC_DMAMUX1_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* DMA interrupt init */
    /* DMA1_Channel1_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    /* USER CODE BEGIN MX_GPIO_Init_1 */
    /* USER CODE END MX_GPIO_Init_1 */

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    HAL_PWREx_EnableVddIO2();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(timestamp_GPIO_Port, timestamp_Pin, GPIO_PIN_SET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(Processing_GPIO_Port, Processing_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(GPIOB, LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin,
                        GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(WW_DETECTED_GPIO_Port, WW_DETECTED_Pin, GPIO_PIN_SET);

    /*Configure GPIO pin : B1_Pin */
    GPIO_InitStruct.Pin = B1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : timestamp_Pin */
    GPIO_InitStruct.Pin = timestamp_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_MEDIUM;
    HAL_GPIO_Init(timestamp_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : Processing_Pin */
    GPIO_InitStruct.Pin = Processing_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_MEDIUM;
    HAL_GPIO_Init(Processing_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pins : LD3_Pin LD2_Pin WW_DETECTED_Pin */
    GPIO_InitStruct.Pin = LD3_Pin|LD2_Pin|WW_DETECTED_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /*Configure GPIO pin : USB_OverCurrent_Pin */
    GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : USB_PowerSwitchOn_Pin */
    GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

    /* USER CODE BEGIN MX_GPIO_Init_2 */
    /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

// interrupt request handler for the I2S DMA
void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
	if (g_i2s_state == FileCapture)
		ee_process_chunk_and_cont_capture(hsai);
	else if( g_i2s_state == Streaming)
		th_process_chunk_and_cont_streaming(hsai);
	else if( g_i2s_state == Stopping)
    {
		printf("Streaming stopped\r\n");
		g_i2s_state = Idle;
	}
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
    /* USER CODE BEGIN Error_Handler_Debug */
    /* User can add his own implementation to report the HAL error
       return state */
    __disable_irq();
    while (1)
    {
    }
/* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
    /* USER CODE BEGIN 6 */
    /* User can add his own implementation to report the file name and
       line number, ex: printf("Wrong parameters value: file %s on line %d\r\n",
                            file, line) */
    /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
