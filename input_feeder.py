'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2,sys,os
import logging as log
import traceback
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        log.basicConfig(level=log.ERROR)
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
        
        
    def load_data(self):
        '''
        Open Input Source File
        '''
        try:
            if (self.input_type!='cam'):
            
                if(not os.path.isfile(self.input_file)):
                    #print(os.path.isfile(self.input_type))
                    raise Exception("Input Source File Path is Not Valid")
            
            if self.input_type=='video':
                self.cap=cv2.VideoCapture(self.input_file)
                
            elif self.input_type=='cam':
                self.cap=cv2.VideoCapture(0)
            else:
                
                self.cap=cv2.imread(self.input_file)
        except Exception as e:
            log.error("Failed to Open Input Source File, Check whether input source path and file is valid.")
            log.error("Exception Error Type:{}".format(str(e)))
            log.error("###Below is traceback for Debug###")
            log.error(traceback.format_exc())
            log.error("Program will Exit!!!")
            sys.exit(0)
        
    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        try:
            if(self.input_type!='image'):
                while True:
                    for _ in range(10):
                        _, frame=self.cap.read()
                    yield frame
            else:
                
                frame =cv2.imread(self.input_file)
                
                   
                yield frame
                
        except Exception as e:
            log.error("Failed to Read Input Source File, Check whether input source is valid image or video and it is not corrupted.")
            log.error("Exception Error Type:{}".format(str(e)))
            log.error("###Below is traceback for Debug###")
            log.error(traceback.format_exc())
            log.error("Program will Exit!!!")
            sys.exit(0)

    def get_fps(self):
        '''
        Get FPS of Video
        '''
        if (self.input_type=='video' or self.input_type=='cam'):
            return int(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            return 0
            
    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

