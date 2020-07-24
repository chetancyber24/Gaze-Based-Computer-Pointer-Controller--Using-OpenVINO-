from face_detection import FaceDetection
from input_feeder import InputFeeder
from facial_landmarks_detection import FaceLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
import os
import sys
import timeit
import cv2
from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
   
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
                        
    parser.add_argument("-fdm", "--face_detect_model", required=True, type=str,
                        help="Path to Face Detect IR Model with model name") 
    parser.add_argument("-flm", "--face_landmarks_model", required=True, type=str,
                        help="Path to Face Landmarks Detect IR Model with model name")
    
    parser.add_argument("-hpm", "--head_pose_model", required=True, type=str,
                        help="Path to Head Pose Estimation IR Model with model name")
    
    parser.add_argument("-gem", "--gaze_estimation_model", required=True, type=str,
                        help="Path to Gaze Estimation IR Model with model name")                    
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-ps", "--perf_stat_lvl",required=False, type=int, default=1,
                        help="Specify Performace Stats (i.e. model load time,infer time). 0 specify no perf stat will be printed on console. 1 specify perf stat for all 4 model's total load and infer time for each frame will be printed along with min, average and max time. 3 specify perf stat for each model's load time and infer time will be printed for each frame "
                        )
                        
    parser.add_argument("-pf", "--perf_stat_file",required=False, type=str, default=None,
                        help="Specify file to write Performace Stats."
                        )
    parser.add_argument("-sf", "--show_frame",required=False, type=int, default=1,
                        help="Specify whether to show frame visually or not, pass 0 not to show frame , pass 1(default) to show frame   "
                        )
                        
    parser.add_argument("-af", "--annot_frame",required=False, type=int, default=1,
                        help="Specify whether to annotate frame with model output or not, pass 0 to disable annotation, pass 1(default) to annotate. Annotation will be disabled if show frame option is disabled."
                        ) 
    parser.add_argument("-mp", "--mouse_prec",required=False, type=str, default='medium',
                        help="Specify Mouse Precision Level, Three level of precision supported {'high', 'low' 'medium'}. Default is 'medium' "
                        )   
                            
    parser.add_argument("-ms", "--mouse_speed",required=False, type=str, default='ultra_fast',
                        help="Specify Mouse Speed , how much time mouse movement should take. Five level of speed supported {'zero_delay':0,'ultra_fast':0.5,'fast':1, 'slow':10, 'medium':5}. Default is 'ultra_fast' "
                        )    
    return parser


    #Inference Pipeline involves first pre-processing input,predict(inference), preprocess output(for next model) , annotate output(if needed) and measure inference time of each model. 
def run_infer_pipeline_face_detection(face_detect,image,prob_threeshold,annot_flag):
     preprocess_image= face_detect.preprocess_input(image)
     start_time = timeit.default_timer()
     fd_output = face_detect.predict(preprocess_image)
     end_time = timeit.default_timer()
     infer_time = end_time-start_time
     face_detected ,bb_coord,annotated_image = face_detect.preprocess_output(fd_output,prob_threeshold=prob_threeshold,annotation_flag=annot_flag)
     return infer_time, face_detected ,bb_coord,annotated_image
     
def run_infer_pipeline_face_landmark_detection(face_lm_detect,image,bb_coord,annotated_image,annot_flag):
    preprocess_flm_image =face_lm_detect.preprocess_input(image,bb_coord)
    start_time = timeit.default_timer()
    flm_output = face_lm_detect.predict(preprocess_flm_image)
    end_time = timeit.default_timer()
    infer_time = end_time-start_time
    left_eye_image,right_eye_image,annotated_image = face_lm_detect.preprocess_output(annotated_image,flm_output,annotation_flag=annot_flag) 
    return infer_time,left_eye_image,right_eye_image,annotated_image

def run_infer_pipeline_head_estimation(head_pose_estimate,image,bb_coord,annotated_image,annot_flag):
    preprocess_head_est_input =head_pose_estimate.preprocess_input(image,bb_coord)
    start_time = timeit.default_timer()
    head_est_output = head_pose_estimate.predict(preprocess_head_est_input)
    end_time = timeit.default_timer()
    infer_time = end_time-start_time
    head_angles,annotated_image  = head_pose_estimate.preprocess_output(annotated_image,head_est_output,annotation_flag=annot_flag)
    return infer_time,head_angles,annotated_image

    
def run_infer_pipeline_gaze_estimation(gaze_estimate,image,left_eye_image,right_eye_image,head_angles,annotated_image,annot_flag):
    input_dict =gaze_estimate.preprocess_input(image,left_eye_image,right_eye_image,head_angles)
    start_time = timeit.default_timer()
    gaze_outputs =gaze_estimate.predict(input_dict)
    end_time = timeit.default_timer()
    infer_time = end_time-start_time
    annotated_image,gaze_output = gaze_estimate.preprocess_output(annotated_image,gaze_outputs,annotation_flag=annot_flag)
    return infer_time,annotated_image,gaze_output

     # Method to print perf stats on console and write stat file.
def log_perf_stat(log_msg ,perf_stat_lvl,perf_stat_file):
    
    if(perf_stat_lvl>0):
        format_text=''
        for _ in range(len(log_msg)):
            format_text =format_text+'#' 
        print(format_text)
        print(log_msg)
        print(format_text)
    if(perf_stat_file is not None):
        perf_stat_file.writelines([format_text+'\n',log_msg+'\n',format_text+'\n'])

    # Method to keep track of min,max and total infer time across frames  for each model  . 
def calculate_historical_infer_stats(current_infer_time,  infer_time_min,infer_time_max,infer_time_total):
    infer_time_total+=current_infer_time
    infer_time_min=infer_time_min if infer_time_min<current_infer_time else current_infer_time
    infer_time_max= infer_time_max if infer_time_max>current_infer_time else current_infer_time
    return  infer_time_min,infer_time_max,infer_time_total
    
def main():
    #Argument parser
    args = build_argparser().parse_args()
    
    # Checking input type image ,video or cam(0)
    image_file_extension_list = ['jpg','jpeg','png','bmp','tiff','gif','webp']
    if(args.input!='0'):
        
        input_file_extension = args.input.split('.')[len(args.input.split('.'))-1]
        
        if input_file_extension in image_file_extension_list:
           input_type ='image'
           
        else:
           input_type ='video'
        input_file = args.input
    
    else:
        input_type ='cam'
        input_file = None
    
    #Set show frame and annotate flag based on argument
    args.show_frame=True if args.show_frame else False
    if(args.show_frame):
        args.annot_frame=True if args.annot_frame else False
    else:
        args.annot_frame=False
    
    feed =InputFeeder(input_type,input_file)
    feed.load_data()
    #fps = feed.get_fps()
    
    
    
    
    #Set Performace Stats Levels based on which performance stats will be printed in console and written in stat file(if provided)  
    perf_stat_lvl=args.perf_stat_lvl
    if(perf_stat_lvl>0 and (args.perf_stat_file is not None)):
        perf_stat_file = open(args.perf_stat_file ,'w')
        perf_stat_file.writelines(["##############################OpenVino Model Performance Stats##############################"])
    else:
        perf_stat_file = None
      
    
    #Initialization of performance counters for all Models
    total_model_load_time = 0
    
    all_model_infer_time =0
    all_model_infer_time_min =99999999999999
    all_model_infer_time_max =0
    all_model_infer_time_avg =0
    all_model_infer_time_total =0
    
    face_detect_infer_time =0
    face_detect_infer_time_min =99999999999999
    face_detect_infer_time_max =0
    face_detect_infer_time_avg =0
    face_detect_infer_time_total=0
    
    face_landmarks_infer_time =0
    face_landmarks_infer_time_min =99999999999999
    face_landmarks_infer_time_max =0
    face_landmarks_infer_time_avg =0
    face_landmarks_infer_time_total=0
    
    head_estimation_infer_time =0
    head_estimation_infer_time_min =99999999999999
    head_estimation_infer_time_max =0
    head_estimation_infer_time_avg =0
    head_estimation_infer_time_total=0
    
    gaze_estimation_infer_time =0
    gaze_estimation_infer_time_min =99999999999999
    gaze_estimation_infer_time_max =0
    gaze_estimation_infer_time_avg =0
    gaze_estimation_infer_time_total=0
    
    #Instantiate Face Detection Class & Load corresponding model
    face_detect=FaceDetection(args.face_detect_model,args.device,args.cpu_extension)
    start_time=timeit.default_timer()
    face_detect.load_model()  
    end_time = timeit.default_timer()
    model_load_time = end_time-start_time # Record Model Load time
    total_model_load_time = total_model_load_time + model_load_time
    log_perf_stat("Face Detection Model Loading Time: {0:.1f}ms".format(model_load_time*1000),perf_stat_lvl,perf_stat_file)
    
    #Instantiate Face Landmarks Detection Class & Load corresponding model
    face_lm_detect = FaceLandmarksDetection(args.face_landmarks_model,args.device,args.cpu_extension)
    start_time=timeit.default_timer()
    face_lm_detect.load_model()  
    end_time = timeit.default_timer()
    model_load_time = end_time-start_time
    total_model_load_time = total_model_load_time + model_load_time
    log_perf_stat("Face Landmarks Detection Model Loading Time: {0:.1f}ms".format(model_load_time*1000),perf_stat_lvl,perf_stat_file)
    
    #Instantiate Head Pose Estimate Class & Load corresponding model
    head_pose_estimate = HeadPoseEstimation(args.head_pose_model,args.device,args.cpu_extension)
    start_time=timeit.default_timer()
    head_pose_estimate.load_model()  
    end_time = timeit.default_timer()
    model_load_time = end_time-start_time
    total_model_load_time = total_model_load_time + model_load_time
    log_perf_stat("Head Estimation Model Loading Time: {0:.1f}ms".format(model_load_time*1000),perf_stat_lvl,perf_stat_file)
    
    #Instantiate Gaze Estimate Class & Load corresponding model
    gaze_estimate = GazeEstimation(args.gaze_estimation_model,args.device,args.cpu_extension)
    start_time=timeit.default_timer()
    gaze_estimate.load_model()  
    end_time = timeit.default_timer()
    model_load_time = end_time-start_time
    total_model_load_time = total_model_load_time + model_load_time
    log_perf_stat("Gaze Estimation Model Loading Time: {0:.1f}ms".format(model_load_time*1000),perf_stat_lvl,perf_stat_file)
    
    #Instantiate Mouse Controller Class and reset mouse pointer to center of screen
    mouse_control = MouseController(args.mouse_prec,args.mouse_speed)
    mouse_control.move_mouse_to_center()
    
    #If Show frame flag is set open frame window
    if(args.show_frame):
        cv2.namedWindow('Output Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output Image', 600,450)
        cv2.moveWindow('Output Image', 600,300)
    
    frame_no=0 
    frame_no_with_face =0
    try:
        for image in feed.next_batch(): # Read frame one by one
            if (image is None):
                break
            if(input_type =='cam'):
                image =cv2.flip(image,1) # In case of cam input, flip image
            frame_no+=1
            image =cv2.resize(image,(1920,1080))
            
            #Run Face Detection Inferrence pipeline(Pre-process Input, Predict &  Pre-process Output)
            face_detect_infer_time,face_detected ,bb_coord,annotated_image = run_infer_pipeline_face_detection(face_detect,image,args.prob_threshold,args.annot_frame)
            
            # Calculate face detection statistical(min,max ,avg) inference time across frames               
            face_detect_infer_time_min,face_detect_infer_time_max,face_detect_infer_time_total =calculate_historical_infer_stats(face_detect_infer_time,face_detect_infer_time_min,face_detect_infer_time_max,face_detect_infer_time_total )
            
            if(perf_stat_lvl>1):#Log frame by frame stats if perf stat level is more than 1
                log_perf_stat("Face Detect Model, Frame No. {} Infer time : {:.2f}ms".format(frame_no,face_detect_infer_time*1000),perf_stat_lvl,perf_stat_file)
            
            if(face_detected):#if face detected run next models inference in pipeline
                frame_no_with_face+=1
                
                #Run Face Landmark Detection Inferrence pipeline(Pre-process Input, Predict &  Pre-process Output)
                face_landmarks_infer_time,left_eye_image,right_eye_image,annotated_image=run_infer_pipeline_face_landmark_detection(face_lm_detect,image,bb_coord,annotated_image,args.annot_frame)
                
                # Calculate face landmark detection statistical(min,max ,avg) inference time across frames 
                face_landmarks_infer_time_min,face_landmarks_infer_time_max,face_landmarks_infer_time_total =calculate_historical_infer_stats(face_landmarks_infer_time,face_landmarks_infer_time_min,face_landmarks_infer_time_max,face_landmarks_infer_time_total )
                if(perf_stat_lvl>1):#Log frame by frame stats if perf stat level is more than 1
                    log_perf_stat("Face LandMarks Detection Model, Frame No. {} Infer time : {:.6f}ms".format(frame_no_with_face,face_landmarks_infer_time*1000),perf_stat_lvl,perf_stat_file)
                    
                
                #Run Head Estimation Inferrence pipeline(Pre-process Input, Predict &  Pre-process Output)
                head_estimation_infer_time,head_angles,annotated_image=run_infer_pipeline_head_estimation(head_pose_estimate,image,bb_coord,annotated_image,args.annot_frame)
                
                # Calculate Head Estimate statistical(min,max ,avg) inference time across frames 
                head_estimation_infer_time_min,head_estimation_infer_time_max,head_estimation_infer_time_total =calculate_historical_infer_stats(head_estimation_infer_time,head_estimation_infer_time_min,head_estimation_infer_time_max,head_estimation_infer_time_total )
                if(perf_stat_lvl>1):#Log frame by frame stats if perf stat level is more than 1
                    log_perf_stat("Head Angles Estimation Model, Frame No. {} Infer time : {:.2f}ms".format(frame_no_with_face,head_estimation_infer_time*1000),perf_stat_lvl,perf_stat_file)
                
                #Run Gaze Estimation Inferrence pipeline(Pre-process Input, Predict &  Pre-process Output)
                gaze_estimation_infer_time,annotated_image,gaze_output =run_infer_pipeline_gaze_estimation(gaze_estimate,image,left_eye_image,right_eye_image,head_angles,annotated_image,args.annot_frame)
                
                # Calculate Gaze Estimate statistical(min,max ,avg) inference time across frames
                gaze_estimation_infer_time_min,gaze_estimation_infer_time_max,gaze_estimation_infer_time_total =calculate_historical_infer_stats(gaze_estimation_infer_time,gaze_estimation_infer_time_min,gaze_estimation_infer_time_max,gaze_estimation_infer_time_total )
                if(perf_stat_lvl>1):
                    log_perf_stat("Gaze Estimation Model, Frame No. {} Infer time : {:.2f}ms".format(frame_no_with_face,gaze_estimation_infer_time*1000),perf_stat_lvl,perf_stat_file)
                
                # Calculate All 4 models total statistical(min,max ,avg) inference time across frames
                all_model_infer_time = face_detect_infer_time + face_landmarks_infer_time +head_estimation_infer_time +  gaze_estimation_infer_time
                
                all_model_infer_time_min,all_model_infer_time_max,all_model_infer_time_total =calculate_historical_infer_stats(all_model_infer_time,all_model_infer_time_min,all_model_infer_time_max,all_model_infer_time_total )
                if(perf_stat_lvl>0):
                    log_perf_stat("All 4 Models Infer Time for Frame No. {} : {:.2f}ms".format(frame_no_with_face,all_model_infer_time*1000),perf_stat_lvl,perf_stat_file)
                
               
                #Set mouse x,y relative movement position base on gaze estimation model output
                move_x = gaze_output[0][0]
                move_y = gaze_output[0][1]
                mouse_control.move(move_x,move_y)
                
                annoatation_text = "Total All Models Inference time : {:.2f}ms".format(all_model_infer_time*1000) 
            else:
                print ("No Face Detected")
                annoatation_text = "No Face Detected"
          
            #If annotation and show frame is True
            if(args.annot_frame):
                annotated_image= cv2.putText(annotated_image,annoatation_text , (20,90), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 4, cv2.LINE_AA)
            if(args.show_frame):    
                cv2.imshow('Output Image' ,annotated_image)
                       
            
            if(input_type =='image'):
                cv2.waitKey(0)
                break
            else:
                if(cv2.waitKey(30)>0):
                    break
                    
            
                   
            
          
        #At last log Summary of performance stats for each models and total all 4 models inference time across all frames
        if(perf_stat_lvl>0):
            
            log_perf_stat("#######Performance Summary Stats#######",perf_stat_lvl,perf_stat_file)
            face_detect_infer_time_avg =face_detect_infer_time_total/frame_no
            log_perf_stat("Face Detect Model Inference Time Summary : Min {:.2f}ms, Avg {:.2f}ms, Max {:.2f}ms".format(face_detect_infer_time_min*1000,face_detect_infer_time_avg*1000,face_detect_infer_time_max*1000),perf_stat_lvl,perf_stat_file)
            
            if(frame_no_with_face!=0):
                face_landmarks_infer_time_avg =face_landmarks_infer_time_total/frame_no_with_face
                log_perf_stat("Face Landmarks Model Inference Time Summary: Min {:.5f}ms, Avg {:.2f}ms, Max {:.2f}ms".format(face_landmarks_infer_time_min*1000,face_landmarks_infer_time_avg*1000,face_landmarks_infer_time_max*1000),perf_stat_lvl,perf_stat_file)
                
                head_estimation_infer_time_avg =head_estimation_infer_time_total/frame_no_with_face
                log_perf_stat("Head Estimation Model Inference Time Summary: Min {:.2f}ms, Avg {:.2f}ms, Max {:.2f}ms".format(head_estimation_infer_time_min*1000,head_estimation_infer_time_avg*1000,head_estimation_infer_time_max*1000),perf_stat_lvl,perf_stat_file)
                
                gaze_estimation_infer_time_avg =gaze_estimation_infer_time_total/frame_no_with_face
                log_perf_stat("Gaze Estimation Model Inference Time Summary: Min {:.2f}ms, Avg {:.2f}ms, Max {:.2f}ms".format(gaze_estimation_infer_time_min*1000,gaze_estimation_infer_time_avg*1000,gaze_estimation_infer_time_max*1000),perf_stat_lvl,perf_stat_file)
                
                all_model_infer_time_avg =all_model_infer_time_total/frame_no_with_face
                log_perf_stat("All 4 Models Total Inference Time Summary: Min {:.2f}ms, Avg {:.2f}ms, Max {:.2f}ms".format(all_model_infer_time_min*1000,all_model_infer_time_avg*1000,all_model_infer_time_max*1000),perf_stat_lvl,perf_stat_file)
           
            
        feed.close()
        if(perf_stat_file is not None):
            perf_stat_file.close()
        cv2.destroyAllWindows() 
    except KeyboardInterrupt: 
           if(not perf_stat_file.closed ):
                perf_stat_file.close()
           if(feed.is_cap_open):
               feed.close() 
               
           cv2.destroyAllWindows() 
           print("Keyboard Interrupt, Exiting!!!")
           sys.exit()
if __name__ == '__main__':
    main()  
    