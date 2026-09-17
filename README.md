# co-bot-intent-detection
The goal of our project is to explore a real-time, low-latency human intent recognition system by deploying a quantized Lite-STGCN on the Qualcomm RB3. The system will leverage a custom heterogeneous hardware pipeline, utilize MediaPipe for pose estimation and offloading to the compute engine (Hexagon DSP) and graph convolutions to the Adreno GPU.

Setup:
1. Install OpenCV with apt (`sudo apt install libopencv-dev`). Make sure `OpenCV_INCLUDE_DIRS` and `OpenCV_LIBS` environment variables are set properly.

2. Download the below version of Qualcomm Neural Processing SDK (QAIRT) from [here](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip).

3. Create `libs/qairt` under the project root and unzip the QAIRT SDK zip file to it. `v2.50.0.260828` folder should be at `PROJECT_ROOT/libs/qairt`.

4. Install cmake with apt

5. Go to project root and run `cmake -S . -B build` (one-time)

6. Build with `cmake --build build -j$(nproc)`

## Model Training Setup
### Essentials
1. Ensure Python 3.10 (this stack does not have wheels for 3.11+)
2. `git clone [https://github.com/uub-buu/co-bot-intent-detection.git](https://github.com/uub-buu/co-bot-intent-detection.git)`
3. `git submodule update --init --recursive -- <path/to/project>`
2. `pip install -r requirements.txt`
3. `pip install -e . --no-deps`
4. `export MPLBACKEND=Agg`

