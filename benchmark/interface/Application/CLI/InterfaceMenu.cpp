#include "InterfaceMenu.hpp"
#include "../IDataSource.hpp"
#include "../ResourceManager.hpp"
#include "usart.h"
#include "baud_config.h"



void CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart)
{
  static const char *name = "InterfaceMenu mutex";
  tx_mutex_create(&CLI::InterfaceMenu::singleton_mutex, (char *)name, 0);
  CLI::InterfaceMenu::InitSingleton(*byte_pool, new IO::Uart(*byte_pool, huart), ResourceManager::GetFileSystem(), ResourceManager::GetWaveSink(), ResourceManager::GetDeviceUnderTest());
}

void CLI_Run()
{
  CLI::InterfaceMenu::GetSingleton().Run();
}

void Record_WW_Detection()
{
  CLI::InterfaceMenu::GetSingleton().RecordOneDetection();
}

void Record_Dutycycle_Start()
{
  CLI::InterfaceMenu::GetSingleton().DutycycleStart();
}
void Record_Dutycycle_Stop()
{
  CLI::InterfaceMenu::GetSingleton().DutycycleStop();
}


namespace CLI
{
  InterfaceMenu *InterfaceMenu::instance = (InterfaceMenu *)TX_NULL;
  TX_MUTEX InterfaceMenu::singleton_mutex = { 0 };

  /**
   * Key-Value pairs that link the command string to functions to execute them
   */
  const menu_command_t InterfaceMenu::menu_struct[] = { {"dut", PassthroughWrapper},
                                                        {"name", NameWrapper},
                                                        {"ls", ListWrapper},
                                                        {"play", PlayWrapper},
														{"setbaud", SetBaudWrapper},
														{"checkbaud", CheckBaudWrapper},
														{"record_detections", RecDetsWrapper},
														{"print_detections", PrintDetsWrapper},
														{"print_dutycycle", PrintDutycycleWrapper},
                                                        {"", DefaultWrapper} };

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::ListWrapper(const std::string &args)
  {
    instance->List(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::NameWrapper(const std::string &args)
  {
    instance->Name(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::PlayWrapper(const std::string &args)
  {
    instance->Play(args);
  }
  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::SetBaudWrapper(const std::string &args)
    {
      instance->SetBaud(args);
    }
  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::CheckBaudWrapper(const std::string &args)
    {
      instance->CheckBaud(args);
    }
  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::RecDetsWrapper(const std::string &args)
  {
    instance->RecordDetections(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::PrintDetsWrapper(const std::string &args)
  {
    instance->PrintDetections(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::PrintDutycycleWrapper(const std::string &args)
  {
    instance->PrintDutycycle(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::DefaultWrapper(const std::string &args)
  {
    instance->Default(args);
  }

  /**
   * Wrap the singleton function in a static function
   */
  void InterfaceMenu::PassthroughWrapper(const std::string &args)
  {
    instance->Passthrough(args);
  }

  /**
   * Create the singleton object.  This is thread safe.
   * @param byte_pool The data pool used to create objects
   * @param uart The host UART
   * @param file_system The file system to interact with
   * @param player The media player
   * @param dut The DUT object
   */
  void InterfaceMenu::InitSingleton(TX_BYTE_POOL &byte_pool, IO::Uart *uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut)
  {
    tx_mutex_get(&singleton_mutex, TX_WAIT_FOREVER);
    if(instance == TX_NULL)
    {
      instance = new InterfaceMenu(byte_pool, *uart, file_system, player, dut);
    }
    tx_mutex_put(&singleton_mutex);
  }

  InterfaceMenu &InterfaceMenu::GetSingleton()
  {
    return *instance;
  }

  /**
   * Constructor for the CLI
   * @param byte_pool The data pool used to create objects
   * @param uart The host UART
   * @param file_system The file system to interact with
   * @param player The media player
   * @param dut The DUT object
   */
  InterfaceMenu::InterfaceMenu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut):
            Menu(byte_pool, uart, menu_struct),
            file_system(file_system),
            player(player),
            dut(dut)
  {

  }

  /**
   * List the files in the given menu
   * @param args The dir to read
   */
  void InterfaceMenu::List(const std::string &args)
  {
    file_system.ListDirectory(args, &queue);
    SendResponse();
    SendEnd();
  }

  /**
   * Return the name of this interface board
   * @param args Ignored
   */
  void InterfaceMenu::Name(const std::string &args)
  {
    SendString("tinyML Enhanced Interface Board");
    SendEndLine();
    SendEnd();
  }

  /**
   * Sends the command to the dut
   * Returns m-dut-passthrough(<<args>>)
   * Then the response from the dut preceeded with "[dut]: "
   *
   * @param args command for the dut
   */
  void InterfaceMenu::Passthrough(const std::string &args)
  {
    static const std::string prefix("[dut]: ");
    SendString("m-dut-passthrough(");
    SendString(args);
    SendString(")");
    SendEndLine();
    SendEnd();
    dut.SendCommand(args, &queue);
    SendResponse(&prefix);
  }

  /**
   * Play a file from the SD card
   * @param args The name of the file to play
   */
  void InterfaceMenu::Play(const std::string &args)
  {
    IDataSource *source = file_system.OpenFile(args);
    Audio::WaveSource wav(*source);
    std::string *msg = new std::string("Playing ");
    msg->append(source->GetName());
    msg->append("\n");
    SendString(msg->c_str());
    player.Play(wav);
    SendEnd();
    delete source;
  }

  /**
   * Set the new baud rate for the board
   */
  void InterfaceMenu::SetBaud(const std::string &args)
  {
      int baud = std::stoi(args);
      SaveBaudRateToFlash(baud);  // Save permanently to Flash

      SendString("set baud to: " + args + "\n");

      SendEnd();
      NVIC_SystemReset();  //Reset the MCU
  }


  /**
   * Check the current baud rate
   */
  void InterfaceMenu::CheckBaud(const std::string &args)
  {
      int currentBaud = huart3.Init.BaudRate;

      SendString("baud is: " + std::to_string(currentBaud) + "\n");

      SendEnd();  // End response
  }




  void InterfaceMenu::RecordDetections(const std::string &args)
  {
	  // Sets the interface board into mode to capture detections,
	  // but the detections are recorded in an interrupt handler,
	  // so we can return here and proceed to playing the wav file

	  __HAL_TIM_SET_COUNTER(&htim2, 0);
	  SendString("Now recording detections");
	  SendEndLine();
	  dut.StartRecordingDetections();
	  SendEnd();
  }

  void InterfaceMenu::PrintDetections(const std::string &args)
  {
	  uint32_t *timestamps = dut.GetDetections();
	  uint32_t num_timestamps = dut.GetNumDetections();

	  SendString("Detection Timestamps (ms)");
	  SendEndLine();
	  for(uint32_t i=0; i<num_timestamps; i++)
	  {
		  SendString(std::to_string(timestamps[i]) + ",");
		  SendEndLine();
	  }
	  SendEnd();
  }

  void InterfaceMenu::PrintDutycycle(const std::string &args)
  {
	  uint32_t *rising_edges = dut.GetDutycycleRisingEdges();
	  uint32_t num_rising_edges = dut.GetNumDutycycleRisingEdges();
	  uint32_t *falling_edges = dut.GetDutycycleFallingEdges();
	  uint32_t num_falling_edges = dut.GetNumDutycycleFallingEdges();

	  // std::string time_str;
	  char time_str[20];

	  SendString("Duty cycle start times (s)");
	  SendEndLine();
	  for(uint32_t i=0; i<num_rising_edges; i++)
	  {
		  std::snprintf(time_str, 20, "%.5f, ",  double(rising_edges[i])/100000.0);
		  SendString(time_str);
		  // SendString(std::to_string(rising_edges[i]) + ", ");
		  if( i+1 % 8 == 0){
			  SendEndLine();
		  }
	  }
	  SendEnd();
  }


  void InterfaceMenu::RecordOneDetection()
  {
	  dut.RecordDetection();
  }

  void InterfaceMenu::DutycycleStart()
  {
	  dut.RecordDutycycleStart();
  }

  void InterfaceMenu::DutycycleStop()
  {
	  dut.RecordDutycycleStop();
  }

}
