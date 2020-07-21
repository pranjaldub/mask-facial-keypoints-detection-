===> FACIAL KEYPOINTS CODE folder ---> contains the training/validation and model architecture code for custom facial keypoint detection


===> MASK DETECTION CODE folder ---> contains two folders
      1) RESNET50 VERSION folder ---> contains mask detection model training and validation code using pretrained RESNET50 architecture
      2) STATE-OF-ART VERSION folder ---> contains mask detection model training and validation code using state-of-art model 


===> INFERENCE CODE (REAL-TIME) folder ---> contain the code for realtime medical mask detection and keypoint prediction using webcam
				      (file given in notebook and script format)
     				      (inferencing from saved models obtained by training above mentioned neural networks)

===> WORKFLOW folder ---> contains two images explaining the workflow of different approaches used.
                          (keypoint detection with pretrained MTCNN and with state-of-art pytorch deep cnn model) 

RESOURCE LINKS ==>
===> DATASETS LINK --->
       1) https://makeml.app/datasets/mask
       2) https://mega.nz/file/l5thGJbY#wQHynfAuq9TVK0OuCVs28itFQYprI8UY6z0APnXOqMg


===> SAVED MODELS NEEDED FOR REAL TIME INFERENCING --->
       1) mask detection saved model ---> https://mega.nz/file/ohUTQQJB#Tcqr7ru2Dpa49JGpLSyXNCgMvu6PU4GEol65qVRiJSI       
       2) facial keypoints saved model --->  https://mega.nz/file/QwUiFAaa#FnAH_dzLp8nNQ6hCCPO1_0IOTjDlOWRbtwOdadpfWgs

===> PRE-REQUISITES --->
	1) Pytorch ---> pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
	2) MTCNN ---> pip install mtcnn
	3) Python ---> guide => https://www.howtogeek.com/197947/how-to-install-python-on-windows/
	
