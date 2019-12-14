# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 15:56
# @Author  : play4fun
# @File    : opencv-with-tkinter.py
# @Software: PyCharm

"""
opencv-with-tkinter.py:
https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
https://www.python-course.eu/tkinter_entry_widgets.php
不需要
pip install image
"""

# import the necessary packages
from tkinter import *
import tkinter.ttk as ttk
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
import imutils
from imutils.video import VideoStream
import threading
import time

def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)

        #  represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged
def showVideo():
    print("hello world")
    thread = threading.Thread(target=videoLoop, args=())
    thread.start()
def stopVideo():
    global threadStop
    threadStop=True
    
def videoLoop():
    # DISCLAIMER:
    # I'm not a GUI developer, nor do I even pretend to be. This
    # try/except statement is a pretty ugly hack to get around
    # a RunTime error that  throws due to threading
    global panelA, panelB, vs, threadStop,rFrame,root
    if not (vs is None):
        return
    try:
        print("[INFO] warming up camera...")
        vs = VideoStream(src=0).start()
        threadStop=False
        rFrame=Frame(root)
        rFrame.grid(row=0,column=1)
        # keep looping over frames until we are instructed to stop
        while not threadStop:
            # grab the frame from the video stream and resize it to
            # have a maximum width of 300 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=600)
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0),
                  (int(0.5 * frame.shape[1]), int(0.8 * frame.shape[0])), (255, 0, 0), 2)
            #  represents images in BGR order; however PIL
            # represents images in RGB order, so we need to swap
            # the channels, then convert to PIL and ImageTk format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # if the panel is not None, we need to initialize it
            if panelA is None:
                panelA = Label(rFrame,image=image)
                panelA.image = image
#                panelA.pack(side="left", padx=10, pady=10)
                panelA.grid(row=0,column=0,padx=50)

            # otherwise, simply update the panel
            else:
                panelA.configure(image=image)
                panelA.image = image
            
            t="hello world"
#            if panelB is None:
#                panelB = Label(text=t)
#                panelB.text = t
#                panelB.pack(side="right", padx=10, pady=10)
#
#            # otherwise, simply update the panel
#            else:
#                panelB.configure(text=t)
#                panelB.text = t
#            panelB=Label(text=t)
#            panelB.pack(side="right", padx=10, pady=10)
        vs.stop()
        vs=None
        panelA=None

    except RuntimeError as e:
        print("[INFO] caught a RuntimeError")
# initialize the window toolkit along with the two image panels
root = Tk()
root.title("手语识别教学软件")
threadStop=False
panelA = None
panelB = None
imglbl=None
rFrame=None
vs=None
selected = IntVar()
checked1=BooleanVar()
checked2=BooleanVar()
checked1.set(True)
checked2.set(True)
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
def readImage(path,size):
    image = cv2.imread(path)
    image=cv2.resize(image,size)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image
    
def showQuestion():
    global imglbl
    lFrame=Frame(root)
    lFrame.grid(row=0,column=0)
    lbl=Label(lFrame,text="含义:",font=("宋体",25),borderwidth=2, relief="groove",width=5,anchor=W,padx=50)
    lbl.grid(row=0,column=0,sticky=W+N,padx=10,pady=10)
    e1=Entry(lFrame,font= "宋体 25 bold",width=14)
    e2=Entry(lFrame,font=("宋体",25),width=14)
    e1.grid(row=0,column=1)
    e2.grid(row=2,column=1)
    image = cv2.imread("pic/a.jpg")
    image=cv2.resize(image,(224,224))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    lbl = Label(lFrame,text="图片:",font=("宋体",25),borderwidth=2, relief="groove",anchor=W,padx=50)
    lbl.grid(row=1,column=0,sticky=W,padx=10)
    imglbl = Label(lFrame,image=image,borderwidth=2, relief="groove",anchor=W,padx=50)
    imglbl.image = image
    imglbl.grid(row=1,column=1,sticky=W,padx=10)
    lbl=Label(lFrame,text="提示:",font=("宋体",25),borderwidth=2, relief="groove",width=5,anchor=W,padx=50)
    lbl.grid(row=2,column=0,sticky=W+S,padx=10,pady=10)
def showNavBar():
    bFrame=Frame(root)
    bFrame.grid(row=1,column=0,columnspan=2)
    btn=Button(bFrame,text="上一题",font=("宋体",15))
    btn.grid(row=0,column=0)
    btn=Button(bFrame,text="下一题",font=("宋体",15))
    btn.grid(row=0,column=2)
    btn=Button(bFrame,text="设置背景",font=("宋体",15))
    btn.grid(row=1,column=0)
    btn=Button(bFrame,text="提交",font=("宋体",15))
    btn.grid(row=1,column=2)
    combo=ttk.Combobox(bFrame)
    combo['values']=(1,2,3,4,"A","B","C")
    combo.grid(row=0,column=1)
    rad1 = Radiobutton(bFrame,text='做手势', value=1, variable=selected, command=showSelectedMode)
    rad2 = Radiobutton(bFrame,text='选图片', value=2, variable=selected, command=showSelectedMode)
    rad3 = Radiobutton(bFrame,text='选含义', value=3, variable=selected, command=showSelectedMode)
    rad1.grid(row=2,column=0)
    rad2.grid(row=2,column=1)
    rad3.grid(row=2,column=2)
    chk=Checkbutton(bFrame,text='显示含义',var=checked1, command=showSelectedMode)
    chk.grid(row=3,column=0)
    chk=Checkbutton(bFrame,text='显示图片',var=checked2, command=showSelectedMode)
    chk.grid(row=3,column=1)
    
def showSelectedMode():
    global rFrame,imglbl
    if not(vs is None):
        stopVideo()
        time.sleep(1)
    if not (rFrame is None):
#        rFrame.destroy()
        print("grid remove")
        rFrame.grid_remove()
    print(selected.get())
    if(selected.get()==3):
        showTextOptions()
    elif(selected.get()==2):
        showPicOptions()
    else:
        showVideo()
    if((not (checked2.get())) and (not (imglbl is None))):
        image=readImage("pic/question.jpg",(224,224))
        imglbl.configure(image=image)
        imglbl.image=image
    else:
        image=readImage("pic/a.jpg",(224,224))
        imglbl.configure(image=image)
        imglbl.image=image
        
    print(checked1.get())
    print(checked2.get())

def showTextOptions():
    global rFrame
    rFrame=Frame(root)
    rFrame.grid(row=0,column=1)
    rad1 = Radiobutton(rFrame,text='字母A',font=("宋体",25), value=1)
    rad2 = Radiobutton(rFrame,text='字母B',font=("宋体",25), value=2)
    rad3 = Radiobutton(rFrame,text='字母C',font=("宋体",25), value=3)
    rad4 = Radiobutton(rFrame,text='字母D',font=("宋体",25), value=4)
    rad1.grid(row=0,column=0,padx=50)
    rad2.grid(row=1,column=0)
    rad3.grid(row=2,column=0)
    rad4.grid(row=3,column=0)
def showPicOptions():
    global rFrame
    rFrame=Frame(root)
    rFrame.grid(row=0,column=1)
    image = readImage("pic/a.jpg",(112,112))
    btn=Radiobutton(rFrame,image=image,value=1)
    btn.image=image
    btn.grid(row=0,column=0)
    image=readImage("pic/b.jpg",(112,112))
    btn=Radiobutton(rFrame,image=image,value=2)
    btn.image=image
    btn.grid(row=0,column=1)
    image=readImage("pic/c.jpg",(112,112))
    btn=Radiobutton(rFrame,image=image,value=3)
    btn.image=image
    btn.grid(row=1,column=0)
    image=readImage("pic/d.jpg",(112,112))
    btn=Radiobutton(rFrame,image=image,value=4)
    btn.image=image
    btn.grid(row=1,column=1)

        
showQuestion()
showNavBar()
btn = Button(root, text="Select an image", command=select_image)
#btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn.grid(row=2,column=0)
btn = Button(root, text="看视频", command=showVideo)
#btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn.grid(row=2,column=1)
btn = Button(root, text="关闭视频", command=stopVideo)
#btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn.grid(row=2,column=2)
# kick off the GUI
root.mainloop()