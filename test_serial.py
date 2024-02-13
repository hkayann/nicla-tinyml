import serial
import struct
import time
import sys
import struct
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# def read_dataset():
#     # Remove columns by their names
#     drops = ['name', 'time']
#     data_path_imu = "./dataset/IMU_10Hz.csv"
#     dataset_imu = pd.read_csv(data_path_imu, on_bad_lines='warn', skiprows=range(1, 100))


#     df_imu = dataset_imu.drop(columns=drops)
#     delay = 120
#     data_arrays = [df_imu.iloc[delay:, i].reset_index(drop=True) for i in range(9)]
#     np_acc_x, np_acc_y, np_acc_z, np_gyro_x, np_gyro_y, np_gyro_z, np_mag_x, np_mag_y, np_mag_z = data_arrays
#     # Combine the features per modality.
#     np_acc = np.stack([np_acc_x, np_acc_y, np_acc_z], axis=1)
#     np_gyro = np.stack([np_gyro_x, np_gyro_y, np_gyro_z], axis=1)
#     np_mag = np.stack([np_mag_x, np_mag_y, np_mag_z], axis=1)
#     # Filter the features with rolling median.
#     # Rolling mean function.
#     def moving_average(arr, window):
#         return np.apply_along_axis(lambda x:
#                                 np.convolve(x, np.ones(window),
#                                 'valid') / window, axis=0, arr=arr)

#     # Set the window size.
#     window_size = 5

#     # Get moving average with given window size.
#     # _f means filtered.
#     np_acc_f = moving_average(np_acc, window_size)
#     np_gyro_f = moving_average(np_gyro, window_size)
#     np_mag_f = moving_average(np_mag, window_size)
#     # Check the lengths.
#     print(f"Lengths: acc-{len(np_acc_f)}, gyro-{len(np_gyro_f)}, mag-{len(np_mag_f)}")

#     # Combine the features.
#     df_filtered = np.concatenate((np_acc_f, np_gyro_f, np_mag_f), axis=1)
#     # Split the dataset into 80%, 20%, 20%.
#     # Calculate split indices.
#     total_rows = df_filtered.shape[0]
#     split_1 = int(total_rows * 0.8)
#     split_2 = int(total_rows * 0.9)

#     # Split the array
#     df_train, df_validation, df_test = np.split(df_filtered, [split_1, split_2])

#     # Convert arrays to dataframes
#     # train_df = pd.DataFrame(df_train)
#     # val_df = pd.DataFrame(df_validation)
#     test_df = pd.DataFrame(df_test)

#     # print(f"Train shape: {df_train.shape}")
#     # print(f"Validation shape: {df_validation.shape}")
#     # print(f"Test shape: {df_test.shape}")
#     # Convert to numpy array
#     np_test = test_df.values
#     # Return that array.
#     return np_test

def read_dataset():

    # Generate quaternion dataframe, perform initial checks.
    # If local the path:
    data_path_IMU = "./datasets/IMU_10Hz.csv" # Freq Drop.

    # Generate the dataframe
    dataset_IMU = pd.read_csv(data_path_IMU, on_bad_lines='warn', skiprows=range(1, 100))

    # Dropping the 'name' and 'time' columns
    dataset_IMU.drop(columns=['name', 'time'], inplace=True)

    # Dropping the first 120 rows
    dataset_IMU = dataset_IMU.iloc[120:]

    # Resetting the index
    dataset_IMU.reset_index(drop=True, inplace=True)

    # Calculate the indices for the train, validation, and test split
    total_length = len(dataset_IMU)
    train_end = int(total_length * 0.6)
    val_end = train_end + int(total_length * 0.2)

    # Split the dataset into train, validation, and test sets
    train_data = dataset_IMU[:train_end]
    validation_data = dataset_IMU[train_end:val_end]
    test_data = dataset_IMU[val_end:]

    # Initialize the scaler with the train data
    scaler_X = StandardScaler().fit(train_data)

    # Transform the test data using the scaler fitted on the training data
    scaled_test_data = scaler_X.transform(test_data)
    # Return the numpy array
    return scaled_test_data

np_test = read_dataset()
row_limit = 45
numFloatsPerRow = 9
PORT = "/dev/cu.usbmodemAA7EAA932"
BAUDRATE = 115200
TIMEOUT = 2

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    time.sleep(2)  # Allow time for the Arduino to reset or for the serial port to become ready

    total_rows = len(np_test)
    start_row = 0

    # Open a text file for writing the echoed data
    with open("tflite_data_real_inference.txt", "w") as file:
        while start_row + row_limit <= total_rows:
            counter = 0
            for row in np_test[start_row:start_row + row_limit]:
                for float_value in row:
                    bytes_to_send = struct.pack('<f', float_value)
                    ser.write(bytes_to_send)
                    print(f"Sent: {float_value}")

                ser.flush()  # Ensure all bytes are sent

                start_time = time.time()
                while ser.inWaiting() < 3:
                    if (time.time() - start_time) > TIMEOUT:
                        print("Timeout waiting for ACK.")
                        sys.exit(1)

                ack = ser.read(3)
                if ack == b'ACK':
                    counter += 1
                    print(f"The acknowledgment {counter} received from Arduino for the row.")
                else:
                    print("ACK not received. Check Arduino script or connection.")
                    sys.exit(1)
                time.sleep(0.1)  # Short delay before sending the next set of floats

            if counter < row_limit:
                print(f"Missing ACK for row {counter+1}.")
                sys.exit(1)

            echoed_floats = []
            for _ in range(numFloatsPerRow):  # Read only the last row of echoed floats
                start_time = time.time()
                while ser.inWaiting() < 4:
                    if (time.time() - start_time) > TIMEOUT:
                        print("Timeout waiting for an echoed float.")
                        sys.exit(1)
                echoed_bytes = ser.read(size=4)
                echoed_float = struct.unpack('<f', echoed_bytes)[0]
                echoed_floats.append(echoed_float)

            # Write the echoed floats to the file
            file.write(','.join(map(str, echoed_floats)) + '\n')

            print(f"Echoed Back: {echoed_floats}")
            print(f"Its length: {len(echoed_floats)}")

            # Move to the next window
            start_row += 1

except serial.SerialException as e:
    print(f"Serial error: {e}")
    sys.exit(1)
except struct.error as e:
    print(f"Struct error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
finally:
    if 'ser' in locals() or 'ser' in globals():
        ser.close()
        print("Serial port closed.")



# # Define the send_rows and receive_echo functions outside of the try block
# def send_rows(start_row):
#     """Sends two rows starting from the given index."""
#     try:
#         for row in range(start_row, start_row + NUM_ROWS):
#             # Convert row to list of floats
#             row_floats = np_test[row, :].tolist()
#             # Pack the floats into binary format
#             data = struct.pack('<' + 'f' * NUM_COLS, *row_floats)
#             ser.write(data)  # Send data to Arduino

#             # Wait for acknowledgment from Arduino
#             while True:
#                 ack = ser.read(1)
#                 if ack == b'\x01':
#                     break
#                 else:
#                     print("Acknowledgment error: Expected 0x01, received", ack.hex())
#                     # Optionally, you can add a delay before retrying
#                     time.sleep(0.1)

#             time.sleep(0.1)  # Short delay between rows
#     except Exception as e:
#         print("An error occurred while sending data:", str(e))


# def receive_echo():
#     """Receives the echoed back data for two rows."""
#     try:
#         expected_bytes = NUM_COLS * 4 * NUM_ROWS  # Expecting bytes for two rows
#         received_bytes = b""
        
#         while len(received_bytes) < expected_bytes:
#             if ser.inWaiting() > 0:
#                 received_bytes += ser.read(ser.inWaiting())
#             else:
#                 print("Waiting for data...")
#                 time.sleep(0.1)
        
#         # Unpack the echoed data back into floats
#         if len(received_bytes) == expected_bytes:
#             echoed_floats = struct.unpack('<' + 'f' * NUM_COLS * NUM_ROWS, received_bytes)
#             return np.array(echoed_floats).reshape((NUM_ROWS, NUM_COLS))
#         else:
#             return None
#     except Exception as e:
#         print("An error occurred while receiving data:", str(e))


# try:
#     # Initialize serial port
#     ser = serial.Serial(PORT, BAUDRATE, timeout=2)
#     time.sleep(2)  # Wait for the connection to be established

#     # Send the first two rows and receive the echo
#     send_rows(0)
#     echoed_rows = receive_echo()
#     ser.flush()
#     if echoed_rows is not None:
#         print("Echoed rows: ", echoed_rows)
#     else:
#         print("Incorrect amount of data received after sending the first two rows")

#     # Slide by one row, send the next two rows, and receive the echo
#     send_rows(1)  # Start from the second row this time
#     echoed_rows = receive_echo()
#     ser.flush()
#     if echoed_rows is not None:
#         print("Echoed rows after sliding by one row: ", echoed_rows)
#     else:
#         print("Incorrect amount of data received after sliding by one row")

#     # Close the serial port
#     ser.close()
# except Exception as e:
#     print("An error occurred:", str(e))