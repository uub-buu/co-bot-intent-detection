# co-bot-intent-detection
The goal of our project is to explore a real-time, low-latency human intent recognition system by deploying a quantized Lite-STGCN on the Qualcomm RB3. The system will leverage a custom heterogeneous hardware pipeline, utilize MediaPipe for pose estimation and offloading to the compute engine (Hexagon DSP) and graph convolutions to the Adreno GPU.

Setup:
1. Install OpenCV with apt (`sudo apt install libopencv-dev`). Make sure `OpenCV_INCLUDE_DIRS` and `OpenCV_LIBS` environment variables are set properly.

2. Download the below version of Qualcomm Neural Processing SDK (QAIRT) from [here](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip). This SDK version was confirmed to work with the Thundercomm Rubik Pi 3 board, with Ubuntu 24.04.

3. Create `libs/qairt` under the project root and unzip the QAIRT SDK zip file to it. `2.22.6.240515` folder should be at `PROJECT_ROOT/libs/qairt`.

4. Install cmake with apt.

5. Install FastRPC library with `sudo apt install -y qcom-fastrpc1`. This installs libcdsprpc.so.

6. Go to project root and run `cmake -S . -B build` (one-time)

7. Build with `cmake --build build -j$(nproc)`

8. Run the program with `./build/cobid <video_file>`

