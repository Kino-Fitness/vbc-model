# vbc-model

In saved folder, which is only visible on your local machine when you run the code, there is a folder for data which is scraped from the google sheets link and downloaded as a csv file, the datafrmaes which is where the post-preprocessing test, train, and val dataframes are stored, and the models which is where the .keras files are.

Check_gpu is a script to ensure that tensorflow can access the GPUs on the current machine

get_data.py preprocesses the data and organizes it into train/val/test splits

utils.py is a script for various functions

vbc.py is the machine learning notebook that uses ensemble learning with bagging to train multiple multi-modal machine learning models (with images channel running through the mobile net v2)

yolov8n.pt is the model used to detect the person in the image and crop right around it to create more accurate data