from openvino.inference_engine import IENetwork, IECore,IEPlugin
import numpy as np
import cv2,math,sys
import logging as log
import traceback
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class GazeEstimation:
    '''
    Class for the Gaze  Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.infer_net = None
        self.input_image =None
        self.cropped_image=None
        self.bb_coord =None
        self.head_angles=None
        self.left_eye_box=None
        self.right_eye_box=None
        self.model_xml = model_name +'.xml'
        self.model_weights = model_name + '.bin'
        self.model=None
        
        self.model_input_head_angles_name = 'head_pose_angles'
        self.model_input_left_eye_name = 'left_eye_image'
        self.model_input_right_eye_name = 'right_eye_image'
        self.model_input_head_angles_shape = None
        self.model_input_left_eye_shape = None
        self.model_input_right_eye_shape = None
        self.model_output_name = None
        self.model_output_shape = None
        log.basicConfig(level=log.ERROR)

    def load_model(self):
        '''
        Load Gaze  Estimation IR Model to Device Plugin.
        '''
        try:
            #Instantiate IECore plugin
            self.plugin = IECore()
            #Load CPU Extension if any
            if((self.extensions is not None) and self.device=='CPU'):
                self.plugin.add_extension(self.extensions,self.device)
            #Read IR Model
            self.model=self.plugin.read_network(model=self.model_xml, weights=self.model_weights)
            #Check if all layers in network supported by device plugin
            self.check_model()
            #Load network to plugin
            self.infer_net=self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)
             #Get model input ,output layer name and shape
            self.model_input_head_angles_shape = self.model.inputs[self.model_input_head_angles_name].shape
            self.model_input_left_eye_shape = self.model.inputs[self.model_input_left_eye_name].shape
            self.model_input_right_eye_shape = self.model.inputs[self.model_input_right_eye_name].shape
            
            self.model_output_name = next(iter(self.model.outputs))
            self.model_output_shape = self.model.outputs[self.model_output_name].shape
        except Exception as e:
            log.error("Failed to Read & Load Gaze Estimation Model, Check whether model path and file is valid.")
            log.error("Exception Error Type:{}".format(str(e)))
            log.error("###Below is traceback for Debug###")
            log.error(traceback.format_exc())
            log.error("Program will Exit!!!")
            sys.exit(0)

    def predict(self,input_dict):
        '''
        Run Inference for Gaze  Estimation Model.
        '''
        outputs = self.infer_net.infer(input_dict)
        return outputs
        

    def check_model(self):
        '''
        Check Supported Layers for Gaze  Estimation Model.
        '''
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device)#Get supported layers by plugin
        layers_unsupported =[] 
        for layer in self.model.layers.keys():
            if(layer not in layers_supported):
               layers_unsupported.append(layer)
        if(len(layers_unsupported) !=0):
            log.error(" Unsupported layers present for Gaze Estimation model . Provide right CPU Extension. Will Exit Now.")
            sys.exit(0)

    def preprocess_input(self, image,left_eye_image,right_eye_image,head_angles,padding=0):
        '''
        Preprocess Input for Gaze  Estimation Model.
        '''
        # Reshape left eye image to input shape of model's input layer       
        left_eye_input_image = cv2.resize(left_eye_image,(self.model_input_left_eye_shape[3], self.model_input_left_eye_shape[2]))
        left_eye_input_image = left_eye_input_image.transpose((2,0,1))
        preprocess_left_eye_image = left_eye_input_image.reshape(1, 3, self.model_input_left_eye_shape[2], self.model_input_left_eye_shape[3])
        
        # Reshape Right eye image to input shape of model's input layer
        right_eye_input_image = cv2.resize(right_eye_image,(self.model_input_right_eye_shape[3], self.model_input_right_eye_shape[2]))
        right_eye_input_image = right_eye_input_image.transpose((2,0,1))
        preprocess_right_eye_image = right_eye_input_image.reshape(1, 3, self.model_input_right_eye_shape[2], self.model_input_right_eye_shape[3])
        
        # Reshape Head Angles array to input shape of model's input layer
        head_angles_np = np.array(head_angles)
        head_angles_input = head_angles_np.reshape(1,3)
        gaze_estimation_input_dict = {self.model_input_head_angles_name:head_angles_input,self.model_input_left_eye_name:left_eye_input_image ,self.model_input_right_eye_name:right_eye_input_image }
        
        return gaze_estimation_input_dict
        

    def preprocess_output(self,annotated_image, outputs,annotation_flag = True):
        '''
        Preprocess Output for Gaze  Estimation Model before feeding it to next model.
        '''
        gaze_vector =outputs[self.model_output_name]
        if( annotation_flag ):
         annoatation_text = "Gaze Vector: X={0:.2f},Y={0:.2f},Z={0:.2f}".format(gaze_vector[0][0],gaze_vector[0][1],gaze_vector[0][2]) 
         annotated_image= cv2.putText(annotated_image,annoatation_text , (20,60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 4, cv2.LINE_AA)
       
        return annotated_image, gaze_vector
        
