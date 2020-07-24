from openvino.inference_engine import IECore,IEPlugin
import cv2,sys
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceLandmarksDetection:
    '''
    Class for the Face Detection Model.
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
        self.model = None
        self.model_input_name = None
        self.model_output_name = None
        self.model_input_shape = None
        self.model_output_shape = None
        
       

    def load_model(self):
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

    def predict(self,preprocess_image):
        
        outputs = self.infer_net.infer({self.model_input_name:preprocess_image})
        return outputs
        

    def check_model(self):
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device)#Get supported layers by plugin
        layers_unsupported =[] 
        for layer in self.model.layers.keys():
            if(layer not in layers_supported):
               layers_unsupported.append(layer)
        if(len(layers_unsupported) !=0):
            print(" Unsupported layers present for Face Landmar Detection model . Provide right CPU Extension. Will Exit Now.")
            sys.exit(0)

    def preprocess_input(self, image,bb_coord,padding=0):
        
       
        self.bb_coord =bb_coord
        self.input_image=image
        # Get Cropped face based on bounding box coordinates from Face detection Model
        cropped_image = image[max(0,bb_coord[1]-padding):min(bb_coord[3]+padding,image.shape[0]-1),max(0,bb_coord[0]-padding):min(bb_coord[2]+padding, image.shape[1]-1)]
        self.cropped_image = cropped_image
       
       # Reshape image to input shape of model's input layer
        cropped_image = cv2.resize(cropped_image,(self.model_input_shape[3], self.model_input_shape[2]))
        cropped_image = cropped_image.transpose((2,0,1))
        preprocess_image = cropped_image.reshape(1, 3, self.model_input_shape[2], self.model_input_shape[3])
        return preprocess_image
        

    def preprocess_output(self, annotated_image,outputs,annotation_flag = True):
        
        
        
        landmarks_det_out = outputs[self.model_output_name]
        
        x1=int(landmarks_det_out[0,0]*self.cropped_image.shape[1])+self.bb_coord[0]#Left eye centre x coord. in original image
        y1=int(landmarks_det_out[0,1]*self.cropped_image.shape[0])+self.bb_coord[1]#Left eye centre y coord.in original image
        x2=int(landmarks_det_out[0,2]*self.cropped_image.shape[1])+self.bb_coord[0]#Right eye centre x coord. in original image
        y2=int(landmarks_det_out[0,3]*self.cropped_image.shape[0])+self.bb_coord[1]#Right eye centre y coord. in original image
       
        eye_coord =[x1,y1,x2,y2]
        image=self.input_image #Original Input image
        
        face_to_eye_width_ratio =3
        eye_box_size = int((self.bb_coord[2]-self.bb_coord[0])/face_to_eye_width_ratio)
        
        #Extract Left Eye and Right Eye Cropped Image
        x_min =0
        y_min =0
        x_max =image.shape[1]
        y_max =image.shape[0]
        left_eye_x1 = max(x_min,eye_coord[0]-int(eye_box_size/2))
        left_eye_x2 = min(x_max,eye_coord[0]+int(eye_box_size/2))
        left_eye_y1 = max(y_min,eye_coord[1]-int(eye_box_size/2))
        left_eye_y2 = min(y_max,eye_coord[1]+int(eye_box_size/2))
        
        right_eye_x1 = max(x_min,eye_coord[2]-int(eye_box_size/2))
        right_eye_x2 = min(x_max,eye_coord[2]+int(eye_box_size/2))
        right_eye_y1 = max(y_min,eye_coord[3]-int(eye_box_size/2))
        right_eye_y2 = min(y_max,eye_coord[3]+int(eye_box_size/2))
        
        
        left_eye_image = image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2]
        
        right_eye_image = image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2]
        
        left_eye_box =[left_eye_x1,left_eye_y1,left_eye_x2,left_eye_y2]
        
        right_eye_box = [right_eye_x1,right_eye_y1,right_eye_x2,right_eye_y2] 
       
        
        if(annotation_flag ):
            annotated_image=cv2.circle(annotated_image, (x1,y1), 5, (0,0,255), -1)
            annotated_image=cv2.circle(annotated_image, (x2,y2), 5, (0,0,255), -1)
            
            annotated_image =cv2.rectangle(annotated_image, (left_eye_box[0],left_eye_box[1]), (left_eye_box[2],left_eye_box[3]), (255,0,0), 2, 8)
            
            annotated_image =cv2.rectangle(annotated_image, (right_eye_box[0],right_eye_box[1]), (right_eye_box[2],right_eye_box[3]), (255,0,0), 2, 8)  
            
        return left_eye_image,right_eye_image,annotated_image
        
