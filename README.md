# Gaze Based Computer Pointer Controller  Using OpenVINO 

## Introduction
Purpose of this project is to control mouse pointer based on eye's gaze using OpenVINO IR models. Before doing Gaze estimation , input video or cam feed need to pass through face detection model, face landmarks, head pose estimation model. Below is flow chart of inference pipeline.
![Diagram showing the flow of data from the input, through the different models, to the mouse controller. ](https://github.com/chetancyber24/Gaze-Based-Computer-Pointer-Controller--Using-OpenVINO-/blob/master/inference_pipeline.png)  
## Project Set Up and Installation
**Prerequisite Software  Requirement**

 1. Intel OpenVINO([\[Download Link\]](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html))([Supported Hardware](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/system-requirements.html)) 
 2. Python 3
 
 **Clone Repo**
 
 To clone this repo, run below command.   
git clone https://github.com/chetancyber24/Gaze-Based-Computer-Pointer-Controller--Using-OpenVINO-/

**Setup Python Virtual Environment**
To setup your own  Python Virtual Environment run below command on command prompt 

    python -m venv  <folder_ path_your_virtual_env>
    
 To activate virtual environment run below command on command prompt
 
 <folder_ path_your_virtual_env>\Scripts\activate.bat

Install dependencies on virtual environment

    pip install -r requirements.txt 

 
**OpenVINO Environment Initialization**
   Run below command  to initialize OpenVINO environment on command prompt 

  

     <openvino_installation_dir>\bin\setupvars.bat

 
   
## Demo
Once you cloned repository and done virtual environment setup and OpenVINO Initialization, you can run below command to run basic demo .

    main.py   -fdm .\models\face-detection-adas-0001\FP32\face-detection-adas-0001 -flm .\models\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 -hpm .\models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001 -gem .\models\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 -i .\demo.mp4 
   You should see video window open with inference output annotated and mouse pointer following person gaze. Below is short demo output.
![Demo Screenshot](https://github.com/chetancyber24/Gaze-Based-Computer-Pointer-Controller--Using-OpenVINO-/blob/master/demo.gif)
## Documentation
To get more details about command line arguments of program, run below command

    python main.py --help
Below its output:

    usage: main.py [-h] -i INPUT -fdm FACE_DETECT_MODEL -flm FACE_LANDMARKS_MODEL
                   -hpm HEAD_POSE_MODEL -gem GAZE_ESTIMATION_MODEL
                   [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]
                   [-ps PERF_STAT_LVL] [-pf PERF_STAT_FILE] [-sf SHOW_FRAME]
                   [-af ANNOT_FRAME] [-mp MOUSE_PREC] [-ms MOUSE_SPEED]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Path to image or video file
      -fdm FACE_DETECT_MODEL, --face_detect_model FACE_DETECT_MODEL
                            Path to Face Detect IR Model with model name
      -flm FACE_LANDMARKS_MODEL, --face_landmarks_model FACE_LANDMARKS_MODEL
                            Path to Face Landmarks Detect IR Model with model name
      -hpm HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                            Path to Head Pose Estimation IR Model with model name
      -gem GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                            Path to Gaze Estimation IR Model with model name
      -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                            MKLDNN (CPU)-targeted custom layers.Absolute path to a
                            shared library with thekernels impl.
      -d DEVICE, --device DEVICE
                            Specify the target device to infer on: CPU, GPU, FPGA
                            or MYRIAD is acceptable. Sample will look for a
                            suitable plugin for device specified (CPU by default)
      -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                            Probability threshold for detections filtering(0.5 by
                            default)
      -ps PERF_STAT_LVL, --perf_stat_lvl PERF_STAT_LVL
                            Specify Performace Stats (i.e. model load time,infer
                            time). 0 specify no perf stat will be printed on
                            console. 1 specify perf stat for all 4 model's total
                            load and infer time for each frame will be printed
                            along with min, average and max time. 3 specify perf
                            stat for each model's load time and infer time will be
                            printed for each frame
      -pf PERF_STAT_FILE, --perf_stat_file PERF_STAT_FILE
                            Specify file to write Performace Stats.
      -sf SHOW_FRAME, --show_frame SHOW_FRAME
                            Specify whether to show frame visually or not, pass 0
                            not to show frame , pass 1(default) to show frame
      -af ANNOT_FRAME, --annot_frame ANNOT_FRAME
                            Specify whether to annotate frame with model output or
                            not, pass 0 to disable annotation, pass 1(default) to
                            annotate. Annotation will be disabled if show frame
                            option is disabled.
      -mp MOUSE_PREC, --mouse_prec MOUSE_PREC
                            Specify Mouse Precision Level, Three level of
                            precision supported {'high', 'low' 'medium'}. Default
                            is 'medium'
      -ms MOUSE_SPEED, --mouse_speed MOUSE_SPEED
                            Specify Mouse Speed , how much time mouse movement
                            should take. Five level of speed supported
                            {'zero_delay':0,'ultra_fast':0.5,'fast':1, 'slow':10,
                            'medium':5}. Default is 'ultra_fast'
**Required Arguments**

 1. Input Source Argument(-i ,--input) - Input can be video file,still image file or webcam feed . To specify cam feed pass 0 to input argument
 2. Model Paths Argument (-fdm,-flm ,-hpm,-gem)- Model path should be specified by this argument. Format is : **<model_path_dir_of_xml_and_bin_file >\model_name**. -fdm,-flm ,-hpm,-gem specify model path for face detection, face landmarks detection ,head pose estimation and gaze estimation model.

**Directory structure**
   

## Benchmarks
Below is benchmarks results for all 4 models across different precision(FP32, FP16 & INT8) with respect to different performance matrix i.e. Model Load Time, Inference Time, FPS and Memory Footprint. These results are obtained on 8th Gen Intel core i5 8350u @1.7GHz, 16GB RAM with 64- Bit Windows 10  Enterprise version. OpenVINO Version is 2020.3.194.

**Model Load Time in Milliseconds**
      
|                                  | FP32   | FP16   | INT8   |
|----------------------------------|--------|--------|--------|
| Face   Detection Model           | 1011.5 | 497.4  | 610.9  |
| Face Landmarks   Detection Model | 160.6  | 155.8  | 152.1  |
| Head Pose   Estimation Model     | 173.2  | 182.4  | 220.5  |
| Gaze   Estimation Model          | 226.5  | 203    | 266.6  |
| All 4 Models                     | 1571.8 | 1038.6 | 1250.1 |

======================================================

**Model Inference  Time in Milliseconds**
|                                  | FP32  | FP16  | INT8  |
|----------------------------------|-------|-------|-------|
| Face   Detection Model           | 18.2  | 16.28 | 12.5  |
| Face Landmarks   Detection Model | 0.63  | 0.66  | 0.63  |
| Head Pose   Estimation Model     | 1.64  | 1.74  | 1.39  |
| Gaze   Estimation Model          | 1.71  | 2.09  | 1.43  |
| All 4   Model                    | 22.18 | 20.77 | 15.95 |

=====================================================

**Model FPS**

|                                  | FP32   | FP16   | INT8   |
|----------------------------------|--------|--------|--------|
| Face   Detection Model           | 54.9   | 61.4   | 80.0   |
| Face Landmarks   Detection Model | 1587.3 | 1515.2 | 1587.3 |
| Head Pose   Estimation Model     | 609.8  | 574.7  | 719.4  |
| Gaze   Estimation Model          | 584.8  | 478.5  | 699.3  |
| All 4   Model                    | 45.1   | 48.1   | 62.7   |

===========================================================

**Model Size in MB**

|                                  | FP32 | FP16 | INT8 |
|----------------------------------|------|------|------|
| Face   Detection Model           | 4    | 2    | 1    |
| Face Landmarks   Detection Model | 0.7  | 0.4  | 0.23 |
| Head Pose   Estimation Model     | 7.3  | 3.6  | 2    |
| Gaze   Estimation Model          | 7.2  | 3.6  | 2    |
| All 4   Model                    | 19.2 | 9.6  | 5.23 |



## Results
As we expect INT8  performance is better than FP32 & FP16 in terms of lower  inference time and higher FPS. Generally expectation is that INT8 performance gain  should be 2x of FP16 and 4x of FP32 performance .  But we are seeing gain of  around 1.4x . Reason could be CPU may not be fully optimized for INT8. Also there is not much difference in FP16 and FP32 performance.


