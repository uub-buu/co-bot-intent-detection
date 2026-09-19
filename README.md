# co-bot-intent-detection
The goal of our project is to explore a real-time, low-latency human intent recognition system by deploying a quantized Lite-STGCN on the Qualcomm RB3. The system will leverage a custom heterogeneous hardware pipeline, utilize MediaPipe for pose estimation and offloading to the compute engine (Hexagon DSP) and graph convolutions to the Adreno GPU.

<img width="1038" height="3945" alt="image" src="https://github.com/user-attachments/assets/82389581-8350-4d5a-ac41-18cfc6f81082" />

## Setup
1. Install OpenCV with apt (`sudo apt install libopencv-dev`). Make sure `OpenCV_INCLUDE_DIRS` and `OpenCV_LIBS` environment variables are set properly.

2. Download the below version of Qualcomm Neural Processing SDK (QAIRT) from [here](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip). This SDK version was confirmed to work with the Thundercomm Rubik Pi 3 board, with Ubuntu 24.04.

3. Create `libs/qairt` under the project root and unzip the QAIRT SDK zip file to it. `2.22.6.240515` folder should be at `PROJECT_ROOT/libs/qairt`.

4. Install cmake with apt.

5. Install FastRPC library with `sudo apt install -y qcom-fastrpc1`. This installs libcdsprpc.so.

6. Go to project root and run `cmake -S . -B build` (one-time)

7. Build with `cmake --build build -j$(nproc)`

8. Run the program with `./build/cobid <video_file>`


## Model Training Setup
### Essentials
1. Ensure Python `3.10` (this stack does not have wheels for 3.11+).

2. `git clone https://github.com/uub-buu/co-bot-intent-detection.git`

3. `git submodule update --init --recursive -- <path/to/project>`

4. `pip install -r requirements.txt`

6. `pip install -e . --no-deps`

7. `export MPLBACKEND=Agg`



## Building STGCN .dlc file

### Python Environment Dependencies
* pip install onnx==1.9.0
* pip install "protobuf<4"
* pip install "numpy<1.24"

### Linux Dependencies
* sudo apt update
* sudo apt install libc++1 libc++abi1 protobuf-compiler

### Patch ONNX Iterable 
PYTHON_SITE_PACKAGES=$(python3 -c "import onnx, os; print(os.path.dirname(onnx.__file__))")

sed -i 's/collections\.Iterable/collections.abc.Iterable/' \
    "${PYTHON_SITE_PACKAGES}/helper.py"

#### Confirm collections.abc is actually imported near the top:
grep -n "^import collections" "${PYTHON_SITE_PACKAGES}/helper.py"
#### If "import collections.abc" isn't already there, add it:
sed -i '/^import collections$/a import collections.abc' \
    "${PYTHON_SITE_PACKAGES}/helper.py"

### Set protobuff runtime to pure-Python mode
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
## Running the model comparison

`python compare_models.py`
