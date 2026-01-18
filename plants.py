import cv2 as cv
import numpy as np 
from keras.models import load_model

model=load_model(r'D:\python\Aiml\projects\plants\Plant_deseases_classification\plants_disease_model.keras')
cap=cv.VideoCapture(0)
img_size=(128,128)
while True:
 success,frame=cap.read()
 if success:
    img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    img=cv.resize(frame,img_size)
    img=img.astype('float32')/255.0
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img,verbose=0)
    deseases=['Apple scab','Apple Black rot','Apple Cedar apple rust','Apple healthy',
              'Blueberry healthy','Cherry (including sour) Powdery mildew','Cherry (including sour) healthy',
              'Corn (maize) Cercospora leaf spot Gray leaf spot','Corn (maize) Common rust ','Corn (maize) Northern Leaf Blight',
              'Corn (maize) healthy','Grape Black rot','Grape Esca (Black Measles)','Grape Leaf blight (Isariopsis Leaf Spot)',
              'Grape healthy','Orange Haunglongbing (Citrus greening)','Peach Bacterial spot','Peach healthy',
              'Pepper, bell Bacterial spot','Pepper, bell healthy','Potato Early blight','Potato Late blight',
              'Potato healthy','Raspberry healthy','Soybean healthy','Squash Powdery mildew',
              'Strawberry Leaf scorch','Strawberry healthy','Tomato Bacterial spot','Tomato Early blight',
              'Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites Two-spotted spider mite',
              'Tomato Target Spot','Tomato Tomato Yellow Leaf Curl Virus','Tomato Tomato mosaic virus',
              'Tomato healthy']
    cv.flip(frame,1,frame)
    cv.putText(frame,f'Prediction:{deseases[np.argmax(pred)]}',(30,40),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow('Plant Disease Detector',frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()