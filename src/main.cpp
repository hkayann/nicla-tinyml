#include <math.h>
#include "tensorflow/lite/core/c/common.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "Arduino_BHY2.h"
#include "Nicla_System.h"

/**
 * Important that the Arduino include comes last if on the Arduino platform, as it has an `abs()` function
 * that will screw with the stdlib abs() function. If need you can use the following lines
 * as well to redeclare the abs() function to be compatible
*/
#include "Arduino.h"
#ifdef ARDUINO
#define abs(x) ((x)>0?(x):-(x))
#endif 

/**
 * Nicla Sense Specs:
 * 512KB Flash / 64KB RAM,
 * 2MB SPI Flash for storage,
 * 2MB QSPI dedicated for BHI260AP.
*/

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Arena size just a round number. The exact arena usage can be determined
// using the RecordingMicroInterpreter.
constexpr int kTensorArenaSize = 4000; // in bytes;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  nicla::begin();
  nicla::leds.begin();
  Serial.begin(9600);
  while(!Serial);
  Serial.println("Initializing ML model...");
  tflite::InitializeTarget();
  Serial.println("TFlite initialized successfully!");

  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  Serial.println("Fetching model...");
  model = tflite::GetModel(g_model);
  Serial.println("Model is fetch!");
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model provided is schema version ");
    Serial.print(model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
  } else {
    Serial.print("Model version: ");
    Serial.println(model->version());
  }
  
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  Serial.println("Pulling all operations...");
  static tflite::MicroMutableOpResolver<2> op_resolver;
  op_resolver.AddFullyConnected();
  op_resolver.AddFullyConnected();
  tflite::MicroProfiler profiler;
  Serial.println("Operations/layers are pulled and set.");
  
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  Serial.println("Interpreter is built.");
  
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    Serial.println(allocate_status);
    return;
  } else {
    Serial.println("AllocateTensors() successed.");
  }

  TFLITE_CHECK_EQ((*interpreter).inputs_size(), 1);
  //(*interpreter).input(0)->data.f[0] = 1.f;
  // Print out the input tensor's details to verify
	// the model is working as expected
  // Obtain pointers to the model's input and output tensors.
	input = interpreter->input(0);
  output = interpreter->output(0);
	Serial.print("Input size: ");
	Serial.println(input->dims->size);
	Serial.print("Input bytes: ");
	Serial.println(input->bytes);
  for (int i = 0; i < input->dims->size; i++) {
		Serial.print("Input dim ");
		Serial.print(i);
		Serial.print(": ");
		Serial.println(input->dims->data[i]);
	}
  for (int i = 0; i < output->dims->size; i++) {
		Serial.print("Output dim ");
		Serial.print(i);
		Serial.print(": ");
		Serial.println(output->dims->data[i]);
	}
}

void loop() {
  nicla::leds.setColor(blue);
  delay(1000);
  
  // Dimensions are (1, 1).
  // Set the input data.
  float input_data[1*1] = {0.1};
  input_data[0] = input_data[0] + 0.1;
  // Copy the data into the input tensor
	for (int i = 0; i < input->bytes; i++) {
		input->data.f[i] = input_data[i];
	}
  // Invoke the model
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		Serial.println("Invoke failed!");
	}
	else {
		Serial.println("Invoke completed!");
	}
  // Print the output data
  float output_data = output->data.f[0];
  Serial.println(output_data);
  nicla::leds.setColor(red);
  delay(1000);
}