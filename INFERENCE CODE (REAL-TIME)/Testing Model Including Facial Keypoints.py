import numpy as np
from PIL import Image
import cv2
from model import Net
import torch
from torchvision import transforms
from mtcnn import MTCNN

def LoadModel(fpath):
    '''
    function to load saved model
    '''
    c = torch.load(fpath, map_location='cpu')
    model = c['model']
    model.load_state_dict(c['state_dict'])
    # as we've to perform testing, we don't need backpropagation so setting 'requires_grad' as false
    for parameter in model.parameters():
        parameter.requires_grad = False
    # model.eval() ->  .eval() does not change any behaviour of gradient calculations , but are used to set specific layers
    #                  like dropout and batchnorm to evaluation mode i.e. dropout layer won't drop activations and 
    #                  batchnorm will use running estimates instead of batch statistics.
    return model.eval()

train_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Initializing file paths for both the models
fpath1 = 'Real-Time Face Mask Detection Model.pth'
fpath2 = 'Facial Keypoints Model.pt'

# Loading the models for testing
model = LoadModel(fpath1)
net = Net()
net.load_state_dict(torch.load(fpath2))
for parameter in net.parameters():
    parameter.requires_grad = False
net.eval()
model_lm = net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = MTCNN()

# Accessing the webcam
cap = cv2.VideoCapture(0)

f = cv2.FONT_HERSHEY_DUPLEX
t = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,255,0)
yellow = (0,155,255)

while (cap.isOpened()):
    # getting the frame in 'frm' and a bool value in 'ret' which is true if a frame is returned
    ret, frm = cap.read()
    if ret == True:
        # converting into grayscale for feature reduction and grayscale images are less computation intensive to operate on
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        col = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        # detecting the faces in the frame returned, it will return the coords of bounding box along with its height and width
        result = detector.detect_faces(col)
        for box in result:
            x, y, w, h = box['box']
            keypoints = box['keypoints']
            # drawing the bounding box based on the coordinates provided by haar_cascade
            cv2.rectangle(frm, (x,y), (x+w,y+h), 2)
            # cropping the portion of image covered by the bounding box
            crp = Image.fromarray(frm,mode = 'RGB')
            #cropped_img = frm[y:y+h, x:x+w]
            cropped_img = crp.crop((x,y,x+w,y+h))
            s = (w*h)/(50000)
            if s<0.5:
                s=0.5
            pil_image = train_transforms(cropped_img)
            image = pil_image.unsqueeze(0)
            # feeding the test cropped image into the model
            result = model(image)
            img = np.array(image)
            img = img[:,0,:,:]
            img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
            result_lm = model_lm(torch.from_numpy(img))
            result_lm = np.array(result_lm)
            result_lm = result_lm*(0.19*h)
            result_lm = result_lm.reshape(68,2)
            result_lm[:,0] += x+(0.28*h)
            result_lm[:,1] += y+(0.49*w)
            _, maximum = torch.max(result.data, 1)
            pred = maximum.item()
            # displaying results based on classification
            if pred == 0:
                cv2.circle(frm, (keypoints['left_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['right_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['nose']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_left']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_right']), 2, yellow, 2)
                (lw,lh), bl = cv2.getTextSize("Correctly Masked", f, s, t)
                cv2.putText(frm, "Correctly Masked", ((int(((w+x)-x-lw)/2)+x),y-10), f, s, green, t)
                cv2.rectangle(frm, (x,y), (x+w,y+h), green, 2)  # green colour rectangle if mask is worn correctly
            elif pred == 1:
                cv2.circle(frm, (keypoints['left_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['right_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['nose']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_left']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_right']), 2, yellow, 2)
                (lw,lh), bl = cv2.getTextSize("Unmasked", f, s, t)
                cv2.putText(frm, "Unmasked", ((int(((w+x)-x-lw)/2)+x),y-10), f, s, red, t)
                cv2.rectangle(frm, (x,y), (x+w,y+h), red, 2)   # red colour rectangle if mask is not being worn
            elif pred == 2:
                cv2.circle(frm, (keypoints['left_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['right_eye']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['nose']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_left']), 2, yellow, 2)
                cv2.circle(frm, (keypoints['mouth_right']), 2, yellow, 2)
                (lw,lh), bl = cv2.getTextSize("Incorrectly Masked", f, s, t)
                cv2.putText(frm, "Incorrectly Masked", ((int(((w+x)-x-lw)/2)+x),y-10), f, s, blue, t)
                cv2.rectangle(frm, (x,y), (x+w,y+h), blue, 2)   # blue colour rectangle if mask is not worn correctly
        cv2.imshow('frame',frm)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # press 'q' to exit
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()