import numpy as np
from scipy.io import wavfile
import time, re, argparse
import serial
from serial.tools import list_ports

g_echo = False

def get_response(port, timeout=None):
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
    end_str = "m-ready"
    bytes_to_transmit = "name%".encode()
    port.write(bytes_to_transmit)
    name = get_response(port, timeout=2.0)
    print(name)
    return name

def extract_feature_array(s):
    match = re.search(r'm-features-\[([^\]]+)\]', s)
    if not match:
        raise ValueError("No 'm-features' array found in the input string.")

    # Split the string of numbers, strip spaces, and convert to int
    number_strings = match.group(1).split(',')
    numbers = [int(n.strip()) for n in number_strings]
    
    return np.array(numbers, dtype=np.int16)

def get_features(wav_data, port, chunk_size=512, num_chunks=None):

    port.write("name%".encode())
    response = get_response(port, timeout=2)
    print(f"Name: {response}")

    total_samples = len(wav_data)
    if num_chunks is None or num_chunks*chunk_size > total_samples:
       # num_chunks is unspecified or chunk_size*num_chunks requests more data than the wav
       num_chunks = total_samples // chunk_size
    elif num_chunks*chunk_size <= total_samples: 
       # the wav has more data than chunk_size*num_chunks requests
       total_samples = num_chunks*chunk_size
    print(f"Sample rate: {samplerate} Hz, Total samples: {total_samples}")

    # Loop through 512-sample chunks
    specgram = None
    for i in range(0, total_samples, chunk_size):
        chunk = wav_data[i:i+chunk_size]
        print(f"\rChunk {i // chunk_size} / {num_chunks}. ", end="")

        num_ints_to_load = min(chunk_size, total_samples-i)
        cmd = f"db load {2*num_ints_to_load}%" # 2 bytes / int16
        if g_echo: print(f"\n<TX> {cmd}")
        port.write(cmd.encode())
        response = get_response(port, timeout=2)
        if g_echo: print(f"<RX> {response}")

        out_str = "db "
        for j, sample in enumerate(chunk):
            # we need to transmit the values as <lsb, msb>
            # 00-01-ff-00 => [256, 255] 
            swapped = ((sample & 0xFF) << 8) | ((sample >> 8) & 0xFF)
            out_str += f"{swapped:04x}"
            if len(out_str) >= 76:
                # DUT has 80-character input buffer
                out_str += "%"
                bytes_to_transmit = out_str.encode()
                if g_echo: print(f"<TX> {out_str}")
                port.write(bytes_to_transmit)
                response = get_response(port, timeout=2)
                if g_echo: print(f"<RX> {response}")
                if "m-ready" not in response:
                    raise RuntimeError("m-ready not receved after db command")
                elif "m-load-done" in response:
                    if g_echo: print(f"DUT done receiving")
                else:
                    if g_echo: print("Continuing")
                out_str = "db "

        out_str += "%"
        if g_echo: print(f"<TX> {out_str}")
        bytes_to_transmit = out_str.encode()
        port.write(bytes_to_transmit)
        response = get_response(port, timeout=2)
        if g_echo: print(f"<RX> {response}")
        if "m-ready" not in response:
            raise RuntimeError("m-ready not receved after db command")
        if "m-load-done" not in response:
            raise RuntimeError("m-load-done not received after transfer")
        out_str = "extract_uart_stream%"
        if g_echo: print(f"<TX> {out_str}")
        bytes_to_transmit = out_str.encode()
        port.write(bytes_to_transmit)
        response = get_response(port, timeout=10)
        feature_vec = extract_feature_array(response)
        if specgram is None:
            specgram = feature_vec.reshape(1, -1)
        else:
            specgram = np.vstack((specgram, feature_vec))
        np.savez("temp_spec.npz", specgram=specgram)
    return specgram

if __name__ == "__main__":
    """
    Takes one positional argument
    """

    parser = argparse.ArgumentParser(prog="stream_wav_uart", description=__doc__)
    parser.add_argument("--wav_file", type=str, help="wav file to stream to DUT", default="sd_card/long_wav_2ch.wav")
    parser.add_argument("--specfile", type=str, help="npz file to save spectrogram to", default="specgram.npz")
    parser.add_argument("--baud", type=int, help="baud rate for communicating with DUT", default=115200)
    parser.add_argument("--verbose", type=str, help="Print out error messag", default="false")
    parser.add_argument("--offset", type=int, help="point in wav file to start extracting", default=0)
    args = parser.parse_args()

    args.offset = 3675136

    if args.verbose.lower() in ["true", "t"]:
        g_echo = True

    # Load WAV file and convert to mono if stereo
    samplerate, data = wavfile.read(args.wav_file)
    if data.ndim > 1:
        print("Stereo detected — taking 1st channel to convert to mono.")
        data = data[:,0]

    data = data[args.offset:]
    # find DUT. Assume that it's the only thing connected to a USB port
    for p in list_ports.comports(): 
        if p.vid:
            dut_port = serial.Serial(p.device, args.baud, timeout=0.1)
            break
    
    # flush out any unretrieved response from previous commands
    dut_port.write("%".encode())
    resp = get_response(dut_port, timeout=1.0)

    print(f"Getting features")
    specgram = get_features(data, dut_port, chunk_size=512)
    
    np.savez(args.specfile, specgram=specgram)
    print(f"Saved spectrogram with shape {specgram.shape} to {args.specfile}")
