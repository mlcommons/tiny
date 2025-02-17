#include "InterfaceMenu.hpp"
#include "../IDataSource.hpp"
#include "../ResourceManager.hpp"

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
  // CLI::InterfaceMenu::GetSingleton().dut.RecordDetection();
  CLI::InterfaceMenu::GetSingleton().RecordOneDetection();
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
														{"record_detections", RecDetsWrapper},
														{"print_detections", PrintDetsWrapper},
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

  void InterfaceMenu::RecordDetections(const std::string &args)
  {

	  std::string *msg = new std::string("Begin recording detections");

	  __HAL_TIM_SET_COUNTER(&htim2, 0);
	  dut.StartRecordingDetections();
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

  void InterfaceMenu::RecordOneDetection()
  {
	  dut.RecordDetection();
  }
}
