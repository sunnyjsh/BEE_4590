import RPi.GPIO as GPIO
import time

class Motors():
    
    def __init__(self):
        self.pests_list = ['housefly']

    def close_door(self):# Import libraries

        # Set GPIO numbering mode5
        GPIO.setmode(GPIO.BOARD)

        # Set pins 11 & 12 as outputs, and define as PWM servo1 & servo2
        GPIO.setup(11,GPIO.OUT)
        servo1 = GPIO.PWM(11,50) # pin 11 for servo1
        GPIO.setup(12,GPIO.OUT)
        servo2 = GPIO.PWM(12,50) # pin 12 for servo2

        # Start PWM running on both servos, value of 0 (pulse off)
        servo1.start(0)
        servo2.start(0)

        # Turn servo1 & 2 to 180
        servo1.ChangeDutyCycle(12)
        servo2.ChangeDutyCycle(2)
        time.sleep(0.5)
        servo1.ChangeDutyCycle(0)
        servo2.ChangeDutyCycle(0)

        #wait for 2 seconds
        time.sleep(2)


        servo1.stop()
        servo2.stop()
        GPIO.cleanup()

    def open_door(self): 
        #turn servo 1 & 2 to 0

        # Set GPIO numbering mode
        GPIO.setmode(GPIO.BOARD)

        # Set pins 11 & 12 as outputs, and define as PWM servo1 & servo2
        GPIO.setup(11,GPIO.OUT)
        servo1 = GPIO.PWM(11,50) # pin 11 for servo1
        GPIO.setup(12,GPIO.OUT)
        servo2 = GPIO.PWM(12,50) # pin 12 for servo2

        # Start PWM running on both servos, value of 0 (pulse off)
        servo1.start(0)
        servo2.start(0)
        servo1.ChangeDutyCycle(2)
        servo2.ChangeDutyCycle(12)
        time.sleep(0.5)
        servo1.ChangeDutyCycle(0)
        servo2.ChangeDutyCycle(0)

        servo1.stop()
        servo2.stop()
        GPIO.cleanup()
    
    # check if the insect detected is in the list of pests_list 
    def check_species(self, name):
        return 0 if name in self.pests_list else 1
    
    def motor_control(self, mode):
        if mode == 0:
            self.close_door()
        elif mode == 1:
            self.open_door()
