# co-bot-intent-detection
The goal of our project is to explore a real-time, low-latency human intent recognition system by deploying a quantized Lite-STGCN on the Qualcomm RB3. The system will leverage a custom heterogeneous hardware pipeline, utilize MediaPipe for pose estimation and offloading to the compute engine (Hexagon DSP) and graph convolutions to the Adreno GPU.

Setup:
1. Install OpenCV with apt (sudo apt install libopencv-dev)
   Make sure OpenCV_INCLUDE_DIRS and OpenCV_LIBS environment variables are set properly.

2. Download the below version of Qualcomm Neural Processing SDK (QAIRT) from https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip.

3. Unzip to libs/qairt. "v2.50.0.260828" folder should be at PROJECT_ROOT/libs/qairt

4. Install cmake with apt

5. Go to project root and run `cmake -S . -B build` (one-time)

6. Build with `cmake --build build -j$(nproc)`
