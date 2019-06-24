import tkinter as tkinter
import os
import cv2
import time
from PIL import Image,ImageFilter
from matplotlib import pyplot as plt
import pandas as panda
from sklearn.tree import DecisionTreeClassifier

data = panda.read_csv("datasets/HandwritingRecognition_MNIST_dataset.csv").as_matrix()
clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:,1:]
train_label = data[0:,0]
clf.fit(xtrain,train_label)

alphabets = {'A':65,'B':66,'C':67,'D':68,'E':69,'F':70,'G':71,'H':72,'I':73,'J':74,'K':75,'L':76,'M':77,'N':78,'O':79,'P':80,'Q':81,'R':82,'S':83,'T':84,'U':85,'V':86,'W':87,'X':88,'Y':89,'Z':90}

def imageprepare(argv):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts)
def btnclick():
    result=[]
    mypath = "tempImageHolder"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    if ent.get() == "":
        btn.configure(text="No Filename")
    else:
        btn.configure(text="Uploading")
        filename = ent.get()
        # The name of the image file
        file_name = os.path.join(
            os.path.dirname(__file__), filename)

    image = cv2.imread(filename)
    method = "left-to-right"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 1
    for contour in sort_contours(contours,method):
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.imwrite("tempImageHolder/"+ str(i) + ".png", image[y:y + h, x:x + w])
        i = i + 1

    time.sleep(10)
    path = os.getcwd()
    path = path + "/tempImageHolder"
    dirs = os.scandir(path)
    for file in dirs:
        head, tail = os.path.split(file)
        x = [imageprepare("tempImageHolder/"+tail)]  # file path here
        print(len(x))  # mnist IMAGES are 28x28=784 pixels
        print(x[0])
        # Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
        newArr = [[0 for d in range(28)] for y in range(28)]
        k = 0
        for i in range(28):
            for j in range(28):
                newArr[i][j] = x[0][k] * 1000
                k = k + 1

        for i in range(28):
            for j in range(28):
                print(newArr[i][j])
                # print(' , ')
            print('\n')

        mat = sum(newArr, [])
        print(clf.predict([mat]))
        l = list(alphabets.keys())[list(alphabets.values()).index(clf.predict([mat]))]
        result.append(l)

        #plt.imshow(newArr, cmap="gray")
        #plt.show()  #Show / plot that image

    lbl5.config(text=''.join(result))

# Instantiate a new GUI Window
window = tkinter.Tk()
window.title("Handwriting Recongnition")
window.geometry("3500x300")
window.configure(background = "#ffffff")

#Defines GUI Elements
lbl = tkinter.Label(window, text="Handwriting Recognition", fg="#383a39", bg="#ffffff", font=("Helvetica", 23))
lbl3 = tkinter.Label(window, text="")
lbl2 = tkinter.Label(window, text="Enter an image's filename and click 'Upload Image'")
ent = tkinter.Entry(window)
btn = tkinter.Button(window, text="Upload Image", command = btnclick)
lbl4 = tkinter.Label(window, text="The prediction as: ")
lbl5 = tkinter.Label(window,text="")


#Packs GUI Elements into window
lbl.pack()
lbl3.pack()
lbl2.pack()
ent.pack()
btn.pack()
lbl4.pack()
lbl5.pack()
window.mainloop()