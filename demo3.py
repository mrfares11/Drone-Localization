#!/usr/bin/env python3
"""
Single-reader MAVLink drone controller with velocity commands
one thread owns the link and dispatches messages to queues
Tested on Pixhawk 6X, MAVLink 2, 57600 baud
"""
import time
import threading
import queue
from datetime import datetime
import os 
#os.environ['MAVLINK20'] = '1'  
from pymavlink import mavutil
import airsim
import math

class OptimizedDroneController:
    def __init__(self, connection_string='udp:127.0.0.1:14550', baud=57600):
        """
        Args
        connection_string : str

        baud : int
            Ignored for UDP/TCP

        """
        self.connection_string = connection_string
        self.baud = baud 
        self.master = None                    # A PLACE HOLDER THAT WIL BE USED TO STORE THE MAVLINK CONNECTION
        self.running = False                  # A BOOLEAN THAT THAT TELLS THE READER THREAD TO KEEP READING DRONE MESSAGES
                                              # OR TO STOP READING THEM

        # THREAD IS A CLASS IN PYTHON, WHILE THREADING IS A MODULE OF THAT CLASS, THREADING IS USED TO ASYNCHRONOUSLY READ 
        # DIFFERENT MESSAGES FROM THE DRONE, AND GENERALLY TO EXECUTE MANY CODE BLOCKS AT THE SAME TIME, HERE IT WILL BE USED
        # TO READ THE MAVLINK MESSAGES FROM THE DRONE IN REAL TIME, AND THE METHODS THAT SEND COMMANDS TO THE DRONE WILL READ
        # THE RELEVANT MESSAGES IN THE THREAD AND ABLE TO EXECUTE THEM IN REAL TIME, WITHOUT TRYING TO REQUEST A SPECIFIC 
        # MESSAGE AT A TIME, FOR EXAMPLE IF I WANT TO ARM, I CAN SEND THE COMMAND AND THEN CONFIRM BY READING FROM THE DRONE
        # THAT IT HAS SUCCESFULLY ARMED, BUT AT THE SAME TIME THE READER IS CHECKING ALWAYS FOR OVVERIDES FROM THE RC TRANSMITTER
        # AND HENCE IT DOESN'T NEED TO CHOOSE WHICH MESSAGE TO READ, SO ARMING OR RC OVVERIDE, IT SEES THEM AT THE SAME TIME
        
        self.reader_thread = None             # THIS IS THE OBJECT OR INSTANCE THAT WILL BE CREATED OUT OF THE THREAD CLASS

        self.telemetry_lock = threading.Lock() # TELEMETRY IS A DICTIONARY THAT HOLDS THE LAST KNOWN VALUES OF THE DRONE TELEMETRY
        # THE LOCK IS AN OBJECT THAT ENSURES THE ABOVE, THAT ONLY ONE THREAD CAN ACCESS THE TELEMETRY DICTIONARY AT A TIME

        self.telemetry = {                    # last-known values
            'altitude_relative': 0.0,
            'armed': False,
            'flight_mode': 'UNKNOWN',
            'battery_voltage': 0.0,
            'gps_fix': 0,
            'satellites': 0,
            'connected': False,
            'local_position': {'x': 0.0, 'y': 0.0, 'z': 0.0}  # NED local position
        }
        self.rc_latest = None                 # last RC_CHANNELS message

        self.ack_queue = queue.Queue()       
        """
        When you tell the drone “arm,” it eventually sends back a small “ACK” message saying “okay” or “nope.” as described above
        The reader thread can’t block waiting for your main code, and your main code can’t block waiting for the reader, so instead they communicate through this queue.
        How it works:
        Reader thread sees an ACK come in → does ack_queue.put(msg).
        Your main code does ack = ack_queue.get(timeout=…) → and reacts when it finds the matching ACK.
        """

        self.flight_modes = {
            0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO',
            4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 7: 'CIRCLE',
            8: 'POSITION', 9: 'LAND', 10: 'OF_LOITER', 11: 'DRIFT',
            13: 'SPORT', 14: 'FLIP', 15: 'AUTOTUNE', 16: 'POSHOLD',
            17: 'BRAKE', 18: 'THROW', 19: 'AVOID_ADSB', 20: 'GUIDED_NOGPS'
        }

    #  Link management                                                      #
    def connect(self, timeout=10):
        try:
            print(f"Connecting on {self.connection_string} @ {self.baud} …")
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                timeout=timeout
            )
            # NOW THIS MASTER OBJECT IS THE MAVLINK CONNECTION THAT SPEAKS THAT MAVLINK PROTOCOL

            print("Waiting for heartbeat …")
            hb = self.master.wait_heartbeat(timeout=timeout)
            if not hb:
                print("✗ No heartbeat")
                return False

            print(f"✓ Connected to system {hb.get_srcSystem()}") # HERE WE GET THE DRONE's SYSTEM ID
            with self.telemetry_lock:
                self.telemetry['connected'] = True

            # request data streams we care about
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                4,   # Hz
                1    # start
            )

            # start single reader thread
            self.running = True
            self.reader_thread = threading.Thread(
                target=self._link_reader, daemon=True
            )
            self.reader_thread.start()
            return True

        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=2)
        if self.master:
            self.master.close()
        with self.telemetry_lock:
            self.telemetry['connected'] = False
        print("Disconnected")

    # --------------------------------------------------------------------- #
    #  Single reader thread                                                 #
    # --------------------------------------------------------------------- #
    def _link_reader(self):
        """Owns recv_match(); dispatches to queues / shared vars."""
        while self.running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue

                mtype = msg.get_type()

                # --- COMMAND_ACK → queue for whoever is waiting --------- #
                if mtype == 'COMMAND_ACK':
                    self.ack_queue.put(msg)

                # --- RC channels → keep the latest ---------------------- #
                elif mtype == 'RC_CHANNELS':
                    self.rc_latest = msg

                # --- Telemetry updates ---------------------------------- #
                elif mtype == 'HEARTBEAT':
                    with self.telemetry_lock:
                        self.telemetry['armed'] = bool(
                            msg.base_mode
                            & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        )
                        self.telemetry['flight_mode'] = self.flight_modes.get(
                            msg.custom_mode, 'UNKNOWN'
                        )

                elif mtype == 'GLOBAL_POSITION_INT':
                    with self.telemetry_lock:
                        self.telemetry['altitude_relative'] = (
                            msg.relative_alt / 1000.0
                        )

                elif mtype == 'SYS_STATUS':
                    with self.telemetry_lock:
                        self.telemetry['battery_voltage'] = (
                            msg.voltage_battery / 1000.0
                        )

                elif mtype == 'GPS_RAW_INT':
                    with self.telemetry_lock:
                        self.telemetry.update(
                            gps_fix=msg.fix_type,
                            satellites=msg.satellites_visible
                        )

                elif mtype == 'LOCAL_POSITION_NED':
                    with self.telemetry_lock:
                        self.telemetry['local_position'] = {
                            'x': msg.x, 'y': msg.y, 'z': msg.z
                        }

            except Exception as e:
                print(f"Reader err: {e}")
        # end while
    # end _link_reader

    # --------------------------------------------------------------------- #
    #  Helper: wait for ACK                                                 #
    # --------------------------------------------------------------------- #
    def _wait_ack(self, cmd_id, timeout=10):
        """Return True if ACK result == ACCEPTED."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                ack = self.ack_queue.get(timeout=remaining)
                if ack.command == cmd_id:
                    return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            except queue.Empty:
                break
        return False

    # --------------------------------------------------------------------- #
    #  High-level commands                                                  #
    # --------------------------------------------------------------------- #
    def arm(self, force=True):
        print("Arming …")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196 if force else 0, 0, 0, 0, 0, 0
        )
        ok = self._wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        print("✓ Armed" if ok else "✗ Arm failed")
        return ok

    def disarm(self, force=False):
        print("Disarming …")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196 if force else 0, 0, 0, 0, 0, 0
        )
        ok = self._wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        print("✓ Disarmed" if ok else "✗ Disarm failed")
        return ok

    def set_mode(self, mode_name):
        mode_map = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'POSHOLD': 16, 'BRAKE': 17, 'GUIDED_NOGPS':20
        }
        if mode_name not in mode_map:
            print(f"Unknown mode {mode_name}")
            return False

        print(f"Changing mode → {mode_name} …")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_map[mode_name], 0, 0, 0, 0, 0
        )
        ok = self._wait_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        print("✓ Mode set" if ok else "✗ Mode change failed")
        return ok

    def takeoff(self, alt, timeout=30):
        if not self.set_mode('GUIDED'):
            return False
        time.sleep(1)
        print(f"Take-off to {alt} m …")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, alt
        )
        ok = self._wait_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        if not ok:
            print("✗ Take-off command rejected")
            return False
        
        print("✓ Take-off command accepted - waiting for altitude...")
        
        # Wait until we reach the target altitude
        tolerance = 0.5  # meters tolerance for altitude reached
        start_time = time.time()
        end_time = start_time + timeout
        
        while time.time() < end_time:
            # Check for manual override
            if self.check_override():
                print("✗ Manual override detected during takeoff")
                return False
            
            # Get current altitude from telemetry
            status = self.get_status()
            current_alt = status.get('altitude_relative', 0.0)
            
            # Check if we've reached the target altitude
            if abs(current_alt - alt) < tolerance:
                print(f"✓ Take-off completed - altitude reached: {current_alt:.1f}m")
                return True
            
            time.sleep(0.1)  # Check at 10Hz
        
        print(f"✓ Take-off timeout - current altitude: {status.get('altitude_relative', 0.0):.1f}m")
        return True  # Return True even on timeout since command was accepted

    def land(self):
        return self.set_mode('LAND')

    # --------------------------------------------------------------------- #
    # Velocity/ Position command method                                         #
    # --------------------------------------------------------------------- #
    def send_pos_vel_command(self, pos, vx, vy, vz, yaw_rate, duration):
        """
        Send velocity command in NED frame for specified duration.
        
        Args:
        ----
        vx : float
            Velocity in North direction (m/s) - positive = forward
            or position in north direction (m)
        vy : float  
            Velocity in East direction (m/s) - positive = right
            or position in east direction (m)
        vz : float
            Velocity in Down direction (m/s) - positive = down
            or position in down direction (m)
        duration : float
            Duration to maintain velocity (seconds)
        
        Returns:
        -------
        bool : True if command executed successfully
        """
        print(f"Velocity/Position command: N={vx:.1f} E={vy:.1f} D={vz:.1f} m/s for {duration}s")
        
        # Ensure we're in GUIDED mode for velocity control
        if not self.set_mode('GUIDED'):
            return False
        
        start_time = time.time()
        end_time = start_time + duration

        if pos:
            type_mask = 1528
            x=vx
            y=vy
            z=vz
            vx = 0
            vy = 0
            vz = 0
        else:
            type_mask = 1479  
            x= 0
            y= 0
            z= 0
        
        if pos:
            # Position command: send once, then wait until position is reached
            print(f"Position command: N={x:.1f} E={y:.1f} D={z:.1f} m (relative)")
            
            # Get current position to calculate target position
            status = self.get_status()
            start_pos = status['local_position']
            target_pos = {
                'x': start_pos['x'] + x,  # Since MAV_FRAME_BODY_OFFSET_NED is relative
                'y': start_pos['y'] + y,
                'z': start_pos['z'] + z
            }
            print(f"Target: N={target_pos['x']:.1f} E={target_pos['y']:.1f} D={target_pos['z']:.1f} m (absolute)")
            
            # Send position command once
            self.master.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms (not used)
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # coordinate frame
                type_mask,  # type_mask for position
                x, y, z,  # x, y, z positions (relative)
                vx, vy, vz,  # velocities (ignored for position command)
                0, 0, 0,  # x, y, z acceleration (ignored)
                0, 0  # yaw, yaw_rate (set to 0 to maintain current yaw)
            )
            
            # Wait until we reach the desired position or timeout
            tolerance = 0.5  # meters tolerance for position reached
            while time.time() < end_time:
                # Check for manual override
                if self.check_override():
                    print("✗ Manual override detected during position command")
                    return False
                
                # Get current position from telemetry
                status = self.get_status()
                current_pos = status['local_position']
                
                # Calculate 3D distance to target
                distance = ((current_pos['x'] - target_pos['x'])**2 + 
                           (current_pos['y'] - target_pos['y'])**2 + 
                           (current_pos['z'] - target_pos['z'])**2)**0.5
                
                if distance < tolerance:
                    print(f"✓ Position reached (distance: {distance:.2f}m)")
                    return True
                
                time.sleep(0.1)  # Check at 10Hz
            
            print(f"✓ Position command timeout - continuing")
            return True
            
        else:
            # Velocity command: send continuously for the duration
            print(f"Velocity command: N={vx:.1f} E={vy:.1f} D={vz:.1f} m/s for {duration}s")
            
            # Send velocity commands at 10Hz for the specified duration
            while time.time() < end_time:
                # Check for manual override
                if self.check_override():
                    print("✗ Manual override detected during velocity command")
                    return False
                
                # Send SET_POSITION_TARGET_LOCAL_NED message with velocity targets
                self.master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms (not used)
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # coordinate frame
                    type_mask,  # type_mask for velocity
                    x, y, z,  # x, y, z positions (ignored for velocity)
                    vx, vy, vz,  # x, y, z velocity in m/s
                    0, 0, 0,  # x, y, z acceleration (ignored)
                    0, yaw_rate  # yaw, yaw_rate (set to 0 to maintain current yaw)
                )
                time.sleep(0.05)  # 20Hz update rate
        
        # Stop the drone by sending zero velocity
        """
        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            1479,  # type_mask
            0, 0, 0,  # positions (ignored)
            0, 0, 0,  # zero velocities to stop
            0, 0, 0,  # accelerations (ignored)
            0, 0  # yaw, yaw_rate (ignored)
        )
        """
        print(f"✓ Velocity command completed")
        return True

    # --------------------------------------------------------------------- #
    #  RC utilities                                                         #
    # --------------------------------------------------------------------- #
    def start_rc_stream(self, rate_hz=5):
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
            rate_hz, 1
        )

    def wait_for_rc7_high(self, threshold=1900, timeout=30):
        print(f"Waiting RC7 > {threshold} …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.rc_latest and getattr(self.rc_latest, 'chan7_raw', 0) > threshold:
                print("✓ RC7 trigger")
                return True
            time.sleep(0.1)
        print("✗ RC7 timeout")
        return False

    def check_override(self):
        if not self.rc_latest:
            return False
        rc6 = getattr(self.rc_latest, 'chan6_raw', 1500)
        # Only trigger override when RC6 is HIGH (> 1700), not when LOW
        # LOW position (< 1300) is normal/safe, HIGH position (> 1700) is override
        if rc6 > 1700:
            print("✗ Manual override detected – aborting")
            return True
        return False

    # --------------------------------------------------------------------- #
    #  Telemetry getters (thread-safe)                                      #
    # --------------------------------------------------------------------- #
    def get_status(self):
        with self.telemetry_lock:
            return dict(self.telemetry)


    # Attitude command method
    def send_attitude_command(self, pitch, yaw, roll, thrust, duration):
        print(f"sending attitude of pitch={pitch} , yaw={yaw} , roll={roll} , thrust={thrust} , for a duration of {duration}")
        if not self.set_mode('GUIDED_NOGPS'):
            return False
        start_time = time.time()
        end_time = start_time + duration
        
        # Convert degrees to radians
        pitch_rad = -(pitch / 180) * 3.14159265359
        yaw_rad = (yaw / 180) * 3.14159265359
        roll_rad = (roll / 180) * 3.14159265359
        
        # Convert Euler angles to quaternion [w, x, y, z]
        # Using standard aerospace convention: roll (x), pitch (y), yaw (z)
        import math
        
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        
        # Quaternion components [w, x, y, z]
        q = [
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy   # z
        ]
        
        type_mask = 7  # Ignore body roll rate, pitch rate, yaw rate
        
        while time.time() < end_time:
            if self.check_override():
                print("✗ Manual override detected during attitude command")
                return False
            
            self.master.mav.set_attitude_target_send(
                0,  # time_boot_ms
                self.master.target_system,
                self.master.target_component,
                type_mask,
                q,  # quaternion [w, x, y, z]
                0, 0, 0,  # body roll rate, pitch rate, yaw rate (ignored due to type_mask)
                thrust
            )
            time.sleep(0.1)
        
        print("attitude command done")
        return True


# ------------------------------------------------------------------------- #
#  AirSim Distance Sensor Functions                                         #
# ------------------------------------------------------------------------- #

def read_distance_sensors_airsim(airsim_ip="172.28.144.1", vehicle_name="Copter", duration=10):
    """
    Connect to AirSim and read distance sensor values in real time at 100ms intervals.
    
    Args:
        airsim_ip (str): IP address of AirSim server (from settings.json LocalHostIp)
        vehicle_name (str): Name of the vehicle in AirSim (from settings.json)
        duration (float): Duration to read sensors in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Connecting to AirSim at {airsim_ip}...")
        
        # Connect to AirSim
        client = airsim.MultirotorClient(ip=airsim_ip)
        client.confirmConnection()
        print("✓ Connected to AirSim")
        
        # Enable API control (optional, depends on your setup)
        client.enableApiControl(True, vehicle_name)
        
        print(f"Reading distance sensors for {duration} seconds at 100ms intervals...")
        print("Distance1 (Right +Y): 15cm forward, 18.5cm right")
        print("Distance2 (Left -Y):  15cm forward, 18.5cm left")
        print("-" * 60)
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            try:
                # Get distance sensor data
                distance_data = client.getDistanceSensorData(vehicle_name=vehicle_name)
                
                # AirSim returns distance data as a dictionary with sensor names as keys
                # The sensor names should match what we defined in settings.json
                distance1 = None
                distance2 = None
                
                # Try to get distance sensor readings
                try:
                    distance1_data = client.getDistanceSensorData("Distance1", vehicle_name)
                    distance1 = distance1_data.distance
                except:
                    distance1 = "N/A"
                
                try:
                    distance2_data = client.getDistanceSensorData("Distance2", vehicle_name)
                    distance2 = distance2_data.distance
                except:
                    distance2 = "N/A"
                
                # Display readings with timestamp
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Distance1 (Right): {distance1:>6} m | Distance2 (Left): {distance2:>6} m")
                
            except Exception as e:
                print(f"Error reading sensors: {e}")
            
            # Wait 100ms before next reading
            time.sleep(0.1)
        
        print("-" * 60)
        print("✓ Distance sensor reading completed")
        
        # Disable API control when done
        client.enableApiControl(False, vehicle_name)
        return True
        
    except Exception as e:
        print(f"✗ AirSim connection error: {e}")
        print("Make sure AirSim is running and the IP address matches your settings.json")
        return False

def test_distance_sensors():
    """
    Test function to read distance sensors from AirSim.
    Uses the IP address from the settings.json configuration.
    """
    # IP address from settings1.json LocalHostIp
    airsim_ip = "172.28.144.1"
    vehicle_name = "Copter"
    
    print("=== AirSim Distance Sensor Test ===")
    print(f"Connecting to AirSim at {airsim_ip}")
    print(f"Vehicle: {vehicle_name}")
    print("Sensors configured:")
    print("  - Distance1: X=0.15m, Y=+0.185m (right side)")
    print("  - Distance2: X=0.15m, Y=-0.185m (left side)")
    print("  - 37cm apart, 15cm forward from center")
    print()
    
    # Read sensors for 10 seconds
    success = read_distance_sensors_airsim(airsim_ip, vehicle_name, duration=10)
    
    if success:
        print("Distance sensor test completed successfully!")
    else:
        print("Distance sensor test failed!")
    
    return success

def get_distance_sensors(airsim_ip="172.28.144.1", vehicle_name="Copter"):
    client = airsim.MultirotorClient(ip=airsim_ip)
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    try:
        distance1 = client.getDistanceSensorData("Distance1", vehicle_name).distance
        distance2 = client.getDistanceSensorData("Distance2", vehicle_name).distance
    except Exception as e:
        print(f"Error getting distance sensors: {e}")
        return None, None
    time.sleep(0.01)
    client.enableApiControl(False, vehicle_name)
    return distance1, distance2


# ------------------------------------------------------------------------- #
#  Main function to run the demo                                          #
# ------------------------------------------------------------------------- #
def main():
    drone = OptimizedDroneController('udp:127.0.0.1:14550', baud=57600)

    try:
        if not drone.connect():
            return

        drone.start_rc_stream()
        if not drone.wait_for_rc7_high():
            return

        if not drone.arm():
            return

        if not drone.takeoff(5):
            return

        # Initialize AirSim connection once
        try:
            airsim_client = airsim.MultirotorClient(ip="172.28.144.1")
            airsim_client.confirmConnection()
            airsim_client.enableApiControl(True, "Copter")
            print("✓ AirSim connected for control loop")
        except Exception as e:
            print(f"✗ AirSim connection failed: {e}")
            return
        
        # Control loop variables
        Integral = 0
        P = 0
        D = 0
        prev_error = 0
        last_time = time.time()
        
        # PID gains
        kp = 0.1
        kd = 0.05
        ki = 0.05
        setpoint_angle = 0
        
        Integral1 = 0
        P1 = 0
        D1 = 0
        prev_error1 = 0

        kp1 = 0.2
        kd1 = 0.05
        ki1 = 0.01
        setpoint_distance = 1
        print("Starting control loop...")
        
        if not drone.set_mode('GUIDED'):
            print("Mode change failed")
            return

        while True:
            loop_start = time.time()
            
            if drone.check_override():
                print("✗ Manual override detected – aborting")
                break
            
            # Get distance sensors (should take ~10ms)
            try:
                distance1_data = airsim_client.getDistanceSensorData("Distance1", "Copter")
                distance2_data = airsim_client.getDistanceSensorData("Distance2", "Copter")
                d1 = distance1_data.distance
                d2 = distance2_data.distance
            except Exception as e:
                print(f"✗ Failed to get distance sensors: {e}")
                break
            
            if d1 is None or d2 is None or d1 <= 0 or d2 <= 0:
                print("✗ Invalid distance sensor readings")
                continue
                
            print(f"Distance1: {d1:.2f} m, Distance2: {d2:.2f} m")
            
            # Calculate yaw angle and distance
            yaw_rad = math.atan((d2-d1)/0.37)  # More robust calculation
            distance = (d1 + d2) / 2
            print(f"Yaw angle: {math.degrees(yaw_rad):.2f} degrees")
            print(f"Distance: {distance:.2f} m")
            
            # PID control calculation
            now = time.time()
            dt = now - last_time
            last_time = now
            
            error = setpoint_angle - yaw_rad
            P = kp * error
            
            if dt > 0:
                Integral += ki * error * dt  # Fixed: Ki -> ki
                D = kd * (error - prev_error) / dt
                prev_error = error
            
            output = P + Integral + D  # Fixed: Use Integral variable
            
            # Limit output to reasonable yaw rate (rad/s)
            output = max(-0.5, min(0.5, output))
            
            print(f"PID: P={P:.3f}, I={Integral:.3f}, D={D:.3f}, Output={output:.3f}")
            
            # Distance PID control calculation
            error1 = setpoint_distance - distance  # Fixed: correct error calculation
            P1 = kp1 * error1
            
            if dt > 0:
                Integral1 += ki1 * error1 * dt
                D1 = kd1 * (error1 - prev_error1) / dt
                prev_error1 = error1
            
            output1 = P1 + Integral1 + D1
            
            # Limit output to reasonable forward velocity (m/s)
            output1 = max(-0.5, min(0.5, output1))
            
            print(f"Distance PID: P={P1:.3f}, I={Integral1:.3f}, D={D1:.3f}, Output={output1:.3f}")
            print(f"Error1: {error1:.3f} (setpoint: {setpoint_distance}, actual: {distance:.2f})")
            
            # Send combined velocity command: forward/back for distance, yaw for angle
            if not drone.send_pos_vel_command(False, 0, 0, 0, -output, 0.1):
                print("✗ Velocity command failed")
                break

            if not drone.send_pos_vel_command(False, -output1, 0, 0, 0, 0.1):
                print("✗ Velocity command failed")
                break

            # Calculate actual loop time
            loop_time = time.time() - loop_start
            print(f"Loop time: {loop_time*1000:.1f}ms")
        
        # Cleanup AirSim connection
        try:
            airsim_client.enableApiControl(False, "Copter")
        except:
            pass

        """
        #Takeoff Sequence Using attitude Commands
        drone.set_mode('GUIDED_NOGPS')

        if not drone.send_attitude_command(5,0,0,0.6,10):
            print('attitude command failed')
            return
        
        
        # Landing Sequence Using attitude Commands
        i=0
        while drone.get_status()['altitude_relative'] > 0.2:
            if drone.check_override():
                return
            if i<=4:
                i+=1
            else:
                i=4
            drone.send_attitude_command(0,0,0,0.6-i/20,0.5)

        print("✓ On the ground")
        # Landing Sequence Using attitude Commands Done





        if not drone.takeoff(10):
            return

        # MODIFIED: Hover for 10 seconds
        print("Hovering for 10 seconds...")
        start = time.time()
        while time.time() - start < 10:
            if drone.check_override():
                return
            stat = drone.get_status()
            print(f"Hovering - Alt {stat['altitude_relative']:.1f} m  "
                  f"Mode {stat['flight_mode']}")
            time.sleep(0.5)

        print("\n=== Starting velocity position command sequence ===")
        # Forward for 2 seconds (positive vx = North/Forward)
        if not drone.send_pos_vel_command(False,2.0, 0.0, 0.0, 5):
            print("✗ Forward velocity command failed")
            return
        
        # Brief pause between commands
        time.sleep(1)
        
        # Backward for 2 seconds (negative vx = South/Backward)  
        if not drone.send_pos_vel_command(False,-2.0, 0.0, 0.0, 5):
            print("✗ Backward velocity command failed")
            return
            
        # Brief pause between commands
        time.sleep(1)
        
        # Right for 2 seconds (positive vy = East/Right)
        if not drone.send_pos_vel_command(False,0.0, 2.0, 0.0, 5):
            print("✗ Right velocity command failed")
            return
            
        # Brief pause between commands
        time.sleep(1)
        
        # Left for 2 seconds (negative vy = West/Left)
        if not drone.send_pos_vel_command(False,0.0, -2.0, 0.0, 5):
            print("✗ Left velocity command failed")
            return

        time.sleep(1)
        if not drone.send_pos_vel_command(False,0.0, 0.0, 2, 1):
            print(" ✗ Downward velocity command failed")
            return
        
        time.sleep(1)

        if not drone.send_pos_vel_command(False,0.0, 0.0, -2, 5):
            print("✗ Upward velocity command failed")
            return
        time.sleep(1)

        if not drone.send_pos_vel_command(True, 10, 0.0, 0, 10):
            print("✗ Position command failed")
            return
        time.sleep(1)
        if not drone.send_pos_vel_command(True, -10, 0.0, 0, 10):
            print("✗ Position command failed")
            return
        time.sleep(1)
        if not drone.send_pos_vel_command(True, 5, 5, -5, 10):
            print("✗ Position command failed")
            return
        time.sleep(1)

        print("✓ Velocity & Position commands sequence completed")
        
        # Brief hover after velocity commands
        print("Final hover before landing...")
        time.sleep(2)
        
        print("Landing …")
        drone.land()s

        # simple landing monitor
        while drone.get_status()['altitude_relative'] > 0.3:
            if drone.check_override():
                return
            time.sleep(0.5)
        print("✓ On the ground")
        """




        drone.disarm(force=True)

    except KeyboardInterrupt:
        print("! Emergency stop")
        drone.land()
        time.sleep(5)
        drone.disarm(force=True)

    finally:
        drone.disconnect()


if __name__ == "__main__":
    #test_distance_sensors()

    main()
