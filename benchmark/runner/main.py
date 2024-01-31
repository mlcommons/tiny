from device_under_test import DUT

from io_manager_enhanced import IOManagerEnhanced
from power_manager import PowerManager
from io_manager import IOManager


def test_power():
  with PowerManager("/dev/tty.usbmodem204C355C55471") as power:
      pass
      for line in power.get_help():
          print(line)
      # power.acquire(500)


def run_dut_test():
  """
  power on
  m-ready
  -mode energy
  imit-done
  m-ready
  :return:
  dut passthrough: profile
  """

def test_dut():
    with DUT("/dev/tty.usbmodem14612303", 9600) as dut:
        print(dut.get_name())
        print(dut.get_model())
        print(dut.get_profile())
        print(dut.timestamp())
        for l in dut.get_help():
            print(l)


def test_io_manager():
  with IOManager('/dev/tty.usbmodem14612201') as io:
      print(io.get_name())
      with DUT(io) as dut:
          print(dut.get_name())
          # print(dut.get_model())
          # print(dut.get_profile())
          # print(dut.timestamp())
          # for l in dut.get_help():
          #     print(l)
      for l in io.get_help():
          print(l)


def test_io_manager_enhanced():
    with IOManagerEnhanced("/dev/tty.usbmodem146403") as io:
        print(io.get_name())
        with DUT(io) as dut:
            print(dut.get_name())
            print(dut.get_model())
            print(dut.get_profile())
            print(dut.timestamp())
            for l in dut.get_help():
                print(l)
        print(io.get_waves())
        print(io.play_wave())
        # waves = io.list_waves()
        # io.play_wave();
        # for w in waves:
        #     print(w)
        #     # io.play_wave(w)


if __name__ == '__main__':
    # try:
    #     test_io_manager()
    # except:
    #     pass
    try:
        test_io_manager_enhanced()
    except:
        pass
    # try:
    #     test_power()
    # except:
    #     pass
    try:
        test_dut()
    except:
        pass
