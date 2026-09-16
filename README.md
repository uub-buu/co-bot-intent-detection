# co-bot-intent-detection
The goal of our project is to explore a real-time, low-latency human intent recognition system by deploying a quantized Lite-STGCN on the Qualcomm RB3. The system will leverage a custom heterogeneous hardware pipeline, utilize MediaPipe for pose estimation and offloading to the compute engine (Hexagon DSP) and graph convolutions to the Adreno GPU

## Setup
### Essentials
1. Ensure Python 3.10 (this stack does not have wheels for 3.11+)
2. `pip install -r requirements.txt`
3. Clone this repo, then from inside it: 
    * `pip install -e . --no-deps`
4. `export MPLBACKEND=Agg`

## Running the model comparison

`python compare_models.p`
