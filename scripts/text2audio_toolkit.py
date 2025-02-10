####################################################################################################
####                        for sinlge video processing (example only)
####################################################################################################

import pyttsx3
engine = pyttsx3.init()


""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
# print (rate)                        #printing current voice rate
engine.setProperty('rate', 130)     # setting up new voice rate


"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
# print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

# engine.say("I will speak this text")
engine.save_to_file('I will speak this text, today is the final day of 2021, see you in the 2022.', 'C:\\Users\\wangx\\OneDrive\\文档\\test.mp3')
engine.runAndWait()






####################################################################################################
####        for dataset-level processing (running on Windows system, visual stuido code)
####################################################################################################

import pyttsx3
import os 
import pdb 

engine = pyttsx3.init()


path = "D:\\TNL2K_JE_longterm_videos\\external\\"
files = os.listdir(path) 

for video in range(len(files)):
    videoName = files[video] 
    print("==>> processing the video: ", videoName) 

    fid = open(path + videoName + "\\language.txt", 'r') 
    text_input = fid.readline()
    
    #### rate of the generated audio file 
    rate_value = 190        ## 80, 100, 125, 150, 170, 190 

    if rate_value == 80: 
        audio_savePath = path + videoName + "\\" + videoName + "_audio_woman_rate080.mp3"
    else:
        audio_savePath = path + videoName + "\\" + videoName + "_audio_woman_rate" + str(rate_value) + ".mp3"

    """ RATE"""
    rate = engine.getProperty('rate')   # getting details of current speaking rate
    # print (rate)                        #printing current voice rate
    engine.setProperty('rate', rate_value)     # setting up new voice rate


    """VOLUME"""
    volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
    # print (volume)                          #printing current volume level
    engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')       #getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

    # engine.say("I will speak this text")
    engine.save_to_file(text_input, audio_savePath)
    engine.runAndWait()

    fid.close()





