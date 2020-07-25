'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui

class MouseController:
    '''
    Class to control mouse pointer movement
    '''
    def __init__(self, precision, speed):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'zero_delay':0,'ultra_fast':0.5,'fast':1, 'slow':10, 'medium':5}

        if(precision in precision_dict.keys()):
           self.precision= precision_dict[precision]
        else:
            self.precision= precision_dict['medium']
        
        if(speed in speed_dict.keys()):
           self.speed= speed_dict[speed]
        else:
            self.speed= speed_dict['ultra_fast']
       
        

    def move(self, x, y,):
        '''
        Method to move pointer to x,y(relative) coordinate.
        '''
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)#
    
    def move_mouse_to_center(self):
        '''
        Method to reset mouse pointer to center of screen.
        '''
        screen_res = pyautogui.size()
        #print("Mouse_speed",self.speed)
        pyautogui.moveTo(int(screen_res[0]/2),int(screen_res[1]/2),0)