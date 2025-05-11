import numpy as np
from scipy.io import wavfile
import time, re, argparse
import serial
from serial.tools import list_ports

import logging
logger = logging.getLogger(__name__)

def get_response(port, timeout=10):
    t0 = time.time()
    response = ""
    end_str = "m-ready"

    while True:
        char = port.read(1).decode()
        if char and char not in '\r\0':
            response = response + char
        if end_str in response:
            break
        if timeout is not None and time.time() - t0 > timeout:
            break
    return response

def get_name(port):
    bytes_to_transmit = "name%".encode()
    port.write(bytes_to_transmit)
    name = get_response(port, timeout=2.0)
    print(name)
    return name

def send_cmd_get_resp(cmd, port, timeout=2, check_echo=True, retries=5):
    logging.info(f"\n<TX> {cmd}")
    i_attempt = 0
    successful = False
    response = ""
    while not successful and i_attempt <= retries:
        try:
            port.write(cmd.encode())
            response = get_response(port, timeout=timeout)
            logging.info(f"<RX> {response}")

            if "m-ready" not in response:
                err_str = f"m-ready not receved after command '{cmd}'"
                logging.error(err_str)
                raise RuntimeError(err_str)
            if check_echo:
                match = re.search(r"Received command:\s*(.+)", response)
                if match:
                    rcvd_cmd = match.group(1).strip()
                    echo_correct = (rcvd_cmd == cmd.rstrip("%"))
                    if not echo_correct:
                        err_str = f"Sent command {cmd}\n. DUT received {rcvd_cmd}"
                        logging.error(err_str)
                        raise RuntimeError(err_str)
                else:
                    err_str = "'Received command:' line not found"
                    logging.error(err_str)
                    raise ValueError(err_str)
        except:
            i_attempt += 1
            info_str = f"Error caught on {cmd}, retry {i_attempt}"
            logging.warning(f"Error caught on {cmd}, retry {i_attempt}")
            print(info_str)
            reset_port(port)
        else:
            successful = True
    if not successful:
        raise RuntimeError("send_cmd_get_resp failed after {i_attempt} retries")
    return response

def extract_feature_array(s):
    match = re.search(r'm-features-\[([^\]]+)\]', s)
    if not match:
        err_str = "No 'm-features' array found in the input string."
        logging.error(err_str)
        raise ValueError(err_str)

    # Split the string of numbers, strip spaces, and convert to int
    number_strings = match.group(1).split(',')
    numbers = [int(n.strip()) for n in number_strings]
    
    return np.array(numbers, dtype=np.int16)

def reset_port(port, timeout=5):
    t0 = time.time()
    port.reset_input_buffer()  # flush buffers on host
    port.reset_output_buffer()
    # flush any partial command in progress on DUT, don't compare txd/rxd cmd

    port.write("%".encode())
    while(True):
        b0 = port.read(1)
        if b0 == b'': # nothing more to read
            break
        if timeout is not None and time.time() - t0 > timeout:
            break


def get_features(wav_data, port, frame_size=512, num_frames=None, offset=0, max_str_len=1028):
    # DUT has 80-character input buffer => max_str_len = 80
    error_count = 0
    bytes_per_msg = (max_str_len - 4)//2 # room for "db " + terminating \0; 2 chars/byte
    bytes_per_msg = 2*(bytes_per_msg//2) # round down to next even number
    sub_frame_len = bytes_per_msg // 2   # number of int16s per "db XXX" msg

    total_samples = len(wav_data)
    # as currently written, this means that if you process C chunks starting at 
    # offset O, you actually need to specify num_chunks=O*chunksize+C
    if num_frames is None or num_frames*frame_size > total_samples:
       # num_chunks is unspecified or chunk_size*num_chunks requests more data than the wav
       num_frames = total_samples // frame_size
    elif num_frames*frame_size <= total_samples: 
       # the wav has more data than chunk_size*num_chunks requests
       total_samples = num_frames*frame_size
    print(f"Sample rate: {samplerate} Hz, Total samples: {total_samples}")

    # Loop through 512-sample chunks
    specgram = None
    for i in range(offset, total_samples, frame_size):
        frame = wav_data[i:i+frame_size]
        print(f"\rChunk {i // frame_size} / {num_frames}. ", end="")
        bytes_txd_this_frame = 0
        cmd = f"db load {2*len(frame)}%" # 2 bytes / int16
        response = send_cmd_get_resp(cmd, port)

        num_sub_frames = np.ceil(2*len(frame)/bytes_per_msg).astype(int)
        for i_sub_frame in range(num_sub_frames):
            sub_frame = frame[sub_frame_len*i_sub_frame:sub_frame_len*(i_sub_frame+1)]
            out_str = "db "
            for j, sample in enumerate(sub_frame):
                # we need to transmit the values as <lsB, msB>
                # 00-01-ff-00 => [256, 255] 
                swapped = ((sample & 0xFF) << 8) | ((sample >> 8) & 0xFF)
                out_str += f"{swapped:04x}"

            out_str += "%"

            try:
                bytes_txd_this_frame += len(sub_frame)*2
                response = send_cmd_get_resp(out_str, port, retries=0)
                
                if "m-load-done" in response:
                    if bytes_txd_this_frame != 2*len(frame):
                        err_str = "DUT done, but frame Tx not complete."
                        logging.error(err_str)
                        raise RuntimeError(err_str)
                    logging.info(f"DUT done receiving (frame transmission complete).")
                else:
                    re_match = re.search(r'(\d+)\s+bytes received', response)
                    if re_match:
                        bytes_rcvd_this_frame = int(re_match.group(1))
                    else:
                        err_str = "'bytes received' not found in response"
                        logging.error(err_str)
                        raise ValueError(err_str)
                    if bytes_rcvd_this_frame != bytes_txd_this_frame:
                        err_str = "Bytes received {bytes_rcvd_this_frame} != bytes transmitted {bytes_txd_this_frame}"
                        logging.error(err_str)
                        raise RuntimeError(err_str)
            except (UnicodeDecodeError, RuntimeError, ValueError) as e:
                previous_bytes_txd = bytes_txd_this_frame - len(sub_frame)*2
                retry_successful = False
                i_attempt = 0

                while not retry_successful and i_attempt < 5:
                    i_attempt += 1
                    print(f"Caught exception.  Re-trying transmission, attempt {i_attempt}")
                    error_count += 1
                    try:
                        reset_port(port)

                        send_cmd_get_resp(f"db setptr {previous_bytes_txd}%", port)
                        response = send_cmd_get_resp(out_str, port)
                        if "m-load-done" in response:
                            if bytes_txd_this_frame != 2*len(frame):
                                err_str = "DUT done, but frame Tx not complete."
                                logging.error(err_str)
                                raise RuntimeError(err_str)
                            logging.info(f"DUT done receiving (frame transmission complete).")
                        else:
                            re_match = re.search(r'(\d+)\s+bytes received', response)
                            if re_match:
                                bytes_rcvd_this_frame = int(re_match.group(1))
                            else:
                                err_str = "'bytes received' not found in response"
                                logging.error(err_str)
                                raise ValueError(err_str)
                            if bytes_rcvd_this_frame != bytes_txd_this_frame:
                                err_str = f"Bytes received {bytes_rcvd_this_frame} != bytes transmitted {bytes_txd_this_frame}"
                                logging.error(err_str)
                                raise RuntimeError(err_str)
                        retry_successful = True
                    except:
                        pass

        if "m-load-done" not in response:
            err_str = "m-load-done not received after transfer"
            logging.error(err_str)
            raise RuntimeError(err_str)
        out_str = "extract_uart_stream%"
        response = send_cmd_get_resp(out_str, port, timeout=10)


        feature_vec = extract_feature_array(response)
        if specgram is None:
            specgram = feature_vec.reshape(1, -1)
        else:
            specgram = np.vstack((specgram, feature_vec))
        np.savez("temp_spec.npz", specgram=specgram)
    print(f"Acquired spectrogram. Caught {error_count} errors")
    return specgram

if __name__ == "__main__":
    """
    Takes one positional argument
    """

    parser = argparse.ArgumentParser(prog="stream_wav_uart", description=__doc__)
    parser.add_argument("--wav_file", type=str, help="wav file to stream to DUT", default="sd_card/long_wav_2ch.wav")
    parser.add_argument("--specfile", type=str, help="npz file to save spectrogram to", default="specgram.npz")
    parser.add_argument("--baud", type=int, help="baud rate for communicating with DUT", default=115200)
    parser.add_argument("--verbose", help="Print out extra debug messages", action='store_true')
    parser.add_argument("--offset", type=int, help="point in wav file to start extracting", default=0)
    args = parser.parse_args()

    logging.basicConfig(filename='stream_wav_uart.log', level=logging.INFO)

    # Load WAV file and convert to mono if stereo
    samplerate, data = wavfile.read(args.wav_file)
    if data.ndim > 1:
        print("Stereo detected — taking 1st channel to convert to mono.")
        data = data[:,0]

    # find DUT. Assume that it's the only thing connected to a USB port
    for p in list_ports.comports(): 
        if p.vid:
            dut_port = serial.Serial(p.device, args.baud, timeout=0.1)
            break
    
    # flush out any unretrieved response from previous commands
    reset_port(dut_port)

    ## If you want to check that the DUT is alive, you can do this
    # response = send_cmd_get_resp("name%", port)
    # print(f"Name: {response}")
    t_start = time.time()
    print(f"Getting features")
    specgram = get_features(data, dut_port, frame_size=512, offset=args.offset)
    t_end = time.time()
    np.savez(args.specfile, specgram=specgram)
    
    print(f"Saved spectrogram with shape {specgram.shape} to {args.specfile}")
    print(f"Completed spectrogram readout in {t_end-t_start} s")
