from openvino.inference_engine import IENetwork, IECore,IEPlugin
import cv2,math,sys
import logging as log
import traceback
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.infer_net = None
        self.input_image =None
        self.cropped_image =None
        self.bb_coord = None
        self.model_xml = model_name +'.xml'
        self.model_weights = model_name + '.bin'
        self.model=None
        self.model_input_name = None
        self.model_output_name = None
        self.model_input_shape = None
        self.model_output_shape = None
        log.basicConfig(level=log.ERROR)

    def load_model(self):
        '''
        Load Head Pose Estimation IR Model to Device Plugin.
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
            self.model_input_name = next(iter(self.model.inputs)) 
            self.model_output_name = next(iter(self.model.outputs))
            self.model_input_shape = self.model.inputs[self.model_input_name].shape
            self.model_output_shape = self.model.outputs[self.model_output_name].shape
        except Exception as e:
            log.error("Failed to Read & Load Head Estimation Model, Check whether model path and file is valid.")
            log.error("Exception Error Type:{}".format(str(e)))
            log.error("###Below is traceback for Debug###")
            log.error(traceback.format_exc())
            log.error("Program will Exit!!!")
            sys.exit(0)

    def predict(self,image):
        '''
        Run Inference for Head Pose Estimation Model.
        '''
        outputs = self.infer_net.infer({self.model_input_name:image})
        return outputs
        

    def check_model(self):
        '''
        Check Supported Layers for Head Pose Estimation Model.
        '''
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device)#Get supported layers by plugin
        layers_unsupported =[] 
        for layer in self.model.layers.keys():
            if(layer not in layers_supported):
               layers_unsupported.append(layer)
        if(len(layers_unsupported) !=0):
            log.error(" Unsupported layers present for Head Pose Estimation model . Provide right CPU Extension. Will Exit Now.")
            sys.exit(0)

    def preprocess_input(self, image,bb_coord,padding=0):
        '''
        Preprocess Input for Head Pose Estimation Model.
        '''
        # Get Cropped face based on bounding box coordinates from Head Pose Estimation Model        
        cropped_image = image[max(0,bb_coord[1]-padding):min(bb_coord[3]+padding,image.shape[0]-1),max(0,bb_coord[0]-padding):min(bb_coord[2]+padding, image.shape[1]-1)]
        self.cropped_image=cropped_image
        self.bb_coord =bb_coord
        # Reshape image to input shape of model's input layer
        cropped_image = cv2.resize(cropped_image,(self.model_input_shape[3], self.model_input_shape[2]))
        cropped_image = cropped_image.transpose((2,0,1))
        preprocess_image = cropped_image.reshape(1, 3, self.model_input_shape[2], self.model_input_shape[3])
        return preprocess_image
        

    def preprocess_output(self, annotated_image,outputs,annotation_flag = True):
        
        '''
        Preprocess Output for Head Pose Estimation Model before feeding it to next model.
        '''
        pitch=outputs["angle_p_fc"][0,0]
        yaw  =outputs["angle_y_fc"][0,0]
        roll =outputs["angle_r_fc"][0,0]        

        angles =[yaw,pitch,roll]
        if( annotation_flag ):
         annoatation_text = "Head Pose Angles in Deg.: P={0:.2f},Y={0:.2f},R={0:.2f}".format(pitch,yaw,roll) 
         annotated_image= cv2.putText(annotated_image,annoatation_text , (20,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 4, cv2.LINE_AA)


        # output_image_with_eye_landmark=cv2.circle(self.original_image, (x2,y2), 5, (0,0,255), -1)
        return angles,annotated_image 

