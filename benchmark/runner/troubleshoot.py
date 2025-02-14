import yaml
from device_manager import DeviceManager
from serial.tools import list_ports

# Load `devices.yaml`
with open("devices.yaml", "r") as file:
    device_defs = yaml.safe_load(file)

# Create DeviceManager and scan for devices
device_manager = DeviceManager(device_defs)
device_manager.scan()

# Print all detected devices
print("\n🔍 Detected Devices from DeviceManager:")
for key, value in device_manager.__dict__.items():
    if isinstance(value, dict):  # Only print stored device dicts
        print(f"🔹 {key}: {value}")

# Try to retrieve PowerManager instance
power_manager = device_manager.get("power", {}).get("instance")

if power_manager:
    print("\n✅ Power Manager Detected!")
    print(f"🔌 Port: {device_manager.get('power', {}).get('port', 'Unknown')}")
else:
    print("\n❌ No Power Manager detected! Check device connections and `devices.yaml`.")

# Additional Debugging: Check if the correct port is assigned
power_port = device_manager.get("power", {}).get("port")
if power_port:
    print(f"\n✅ Detected Power Manager Port: {power_port}")
else:
    print("\n❌ No port assigned for Power Manager.")

# List all available serial devices for debugging
print("\n🔍 Listing Available Serial Ports:")
ports = list(list_ports.comports())
for port in ports:
    print(f"Device: {port.device}, Description: {port.description}, VID: {port.vid}, PID: {port.pid}")
