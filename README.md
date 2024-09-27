# jjjjjgggj.github.io
paper code
##If you need to use this code for training!!!!!
###1.you need to create a dataset folder and then upload your data. 
###2.if you need to conduct a 2X Super-resolution experiment, modify config/sr_MDDPM_mri_64_128.json. In this configuration file, input the corresponding dataset path and image resolution.
###3.if you need to conduct a 4X Super-resolution experiment, modify config/sr_MDDPM_mri_32_128.json. In this configuration file, input the corresponding dataset path and image resolution.
#######After configuring the JSON file, fill it in with sr2.py. You can start training now!
##If you need to use this code for testing!!!!!
###1.Fill in the saved model path into JSON and modify the phase in JSON.
#######After configuring the Json file, fill it in with infer.py. You can start testing now!


####About dataset!!
#The Alzheimer MRI Preprocessed Dataset and the Knee MRI dataset that support the findings of this study are openly available in Kaggle at https://www.kaggle.com and the T1-weighted brain MRI dataset that support
the findings of this study are openly available in Openneuro data sharing platform at https://openneuro.org/datasets/ds002785.
