from openvino.inference_engine import IECore,IEPlugin
import cv2,sys
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
       
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.infer_net = None
        self.input_image =None
        self.original_image =None
        self.model_xml = model_name +'.xml'
        self.model_weights = model_name + '.bin'
        self.model=None
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
    
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device) #Get supported layers by plugin
        layers_unsupported =[] 
        for layer in self.model.layers.keys():
            if(layer not in layers_supported):
               layers_unsupported.append(layer)
        if(len(layers_unsupported) !=0):
            print(" Unsupported layers present for Face Detection model . Provide right CPU Extension. Will Exit Now.")
            sys.exit(0)
        

    def preprocess_input(self, image):
        
        self.original_image = image
        # Reshape image to input shape of model's input layer
        input_image = cv2.resize(image,(self.model_input_shape[3], self.model_input_shape[2]))
        input_image = input_image.transpose((2,0,1))
        preprocess_image = input_image.reshape(1, 3, self.model_input_shape[2], self.model_input_shape[3])
        return preprocess_image
        

    def preprocess_output(self, outputs,prob_threeshold=0.5,annotation_flag = True):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x1=None
        y1=None
        x2=None
        y2=None
        face_detected = False
        output_image_with_bb =None
        current_max_prob_threeshold=prob_threeshold
        obj_det_out = outputs[self.model_output_name]
        for i in range(obj_det_out.shape[2]):#Read confidence  and label iteratively for each detected objects
            confidence =  obj_det_out[0,0,i,2]  
            label=obj_det_out[0,0,i,1]
            #print(obj_det_out[0,0,i,:])
            
            if ((label==1) and (confidence> current_max_prob_threeshold)): #Check if person(label=1) is detected and and confidence is greater than prob. threshold
                current_max_prob_threeshold =confidence
                face_detected = True
                #print("label & Confidence %d ,%d",(label,confidence))
                x1=int(obj_det_out[0,0,i,3]*self.original_image.shape[1])
                y1=int(obj_det_out[0,0,i,4]*self.original_image.shape[0])
                x2=int(obj_det_out[0,0,i,5]*self.original_image.shape[1])
                y2=int(obj_det_out[0,0,i,6]*self.original_image.shape[0])
        
        bb_coord =[x1,y1,x2,y2]
        annotated_image =self.original_image.copy()
        if(face_detected and annotation_flag ):
            annotated_image=cv2.rectangle(annotated_image, (x1,y1), (x2,y2), (0,0,255),4 , 8) #int(round(self.model_input_shape[2]/150))
        
        
            
        return face_detected, bb_coord,annotated_image
        
