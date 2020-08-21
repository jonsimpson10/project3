from flask import Flask, render_template
import requests
import pandas as pd
import requests


# Create an instance of Flask
app = Flask(__name__)

# Route to render index.html template using data from Mongo
@app.route("/")
def home():
    # final_string = ''
    def plate_reader(plate_pic):
        # remove warning message
        # remove warning message
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # required library
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from local_utils import detect_lp
        from os.path import splitext,basename
        from keras.models import model_from_json
        from sklearn.preprocessing import LabelEncoder
        import glob


        def load_model(path):
            try:
                path = splitext(path)[0]
                with open('%s.json' % path, 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json, custom_objects={})
                model.load_weights('%s.h5' % path)
                print("Loading model successfully...")
                return model
            except Exception as e:
                print(e)

                
        wpod_net_path = "wpod-net.json"
        wpod_net = load_model(wpod_net_path)

        def preprocess_image(image_path,resize=False):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            if resize:
                img = cv2.resize(img, (255,255))
            return img

        def get_plate(image_path, Dmax=608, Dmin=256):
            vehicle = preprocess_image(image_path)
            ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)
            _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
            return vehicle, LpImg, cor

        # test_image_path = "Plate_examples/usa_car_plate.jpg"
        test_image_path = "Plate_examples/nick_test.jpg"
        vehicle, LpImg,cor = get_plate(test_image_path)

        # fig = plt.figure(figsize=(12,6))
        # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
        # fig.add_subplot(grid[0])
        # plt.axis(False)
        # plt.imshow(vehicle)
        # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
        # fig.add_subplot(grid[1])
        # plt.axis(False)
        # plt.imshow(LpImg[0])

        if (len(LpImg)): #check if there is at least one license image
            # Scales, calculates absolute values, and converts the result to 8-bit.
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            
            # convert to grayscale and blur the image
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),0)
            
            # Applied inversed thresh_binary 
            binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2) 
            
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            
        # # visualize results    
        # fig = plt.figure(figsize=(12,7))
        # plt.rcParams.update({"font.size":18})
        # grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
        # plot_image = [plate_image, gray, blur, binary,thre_mor]
        # # plot_image = [ gray]
        # plot_name = ["gray","blur","binary","dilation", 'thre_mor']

        # for i in range(len(plot_image)):
        #     fig.add_subplot(grid[i])
        #     plt.axis(False)
        #     plt.title(plot_name[i])
        #     if i ==0:
        #         plt.imshow(plot_image[i])
        #     else:
        #         plt.imshow(plot_image[i],cmap="gray")

        # plt.savefig("threshding.png", dpi=300)

        # Create sort_contours() function to grab the contour of each digit from left to right
# Create sort_contours() function to grab the contour of each digit from left to right
        def sort_contours(cnts,reverse = False):
            i = 0
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][i], reverse=reverse))
            return cnts

        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 20, 50

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.4: # Select contour which has the height larger than 40% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        print("Detect {} letters...".format(len(crop_characters)))
        # fig = plt.figure(figsize=(10,6))
        # plt.axis(False)
        # plt.imshow(test_roi)
        #plt.savefig('grab_digit_contour.png',dpi=300)

        # fig = plt.figure(figsize=(14,4))
        # grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

        # for i in range(len(crop_characters)):
        #     fig.add_subplot(grid[i])
        #     plt.axis(False)
        #     plt.imshow(crop_characters[i],cmap="gray")
        #plt.savefig("segmented_leter.png",dpi=300)  


        # Load model architecture, weight and labels
        json_file = open('MobileNets_character_recognition.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("License_character_recognition_weight.h5")
        print("[INFO] Model loaded successfully...")

        labels = LabelEncoder()
        labels.classes_ = np.load('license_character_classes.npy')
        print("[INFO] Labels loaded successfully...")
        # plt.savefig("threshding.png", dpi=300)

        # pre-processing input images and pedict with model
        def predict_from_model(image,model,labels):
            image = cv2.resize(image,(80,80))
            image = np.stack((image,)*3, axis=-1)
            prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
            return prediction
            
        # fig = plt.figure(figsize=(15,3))
        # cols = len(crop_characters)
        # grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

        final_string = ''

        for i,character in enumerate(crop_characters):
            # fig.add_subplot(grid[i])
            title = np.array2string(predict_from_model(character,model,labels))
            # plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            final_string+=title.strip("'[]")
            # plt.axis(False)
            # plt.imshow(character,cmap='gray')


        print("Achieved result: ", final_string)

        # plt.savefig('final_result.png', dpi=300)
        print(type(final_string))
        return final_string
    # plate_reader('Plate_examples/nissan-gtr-nismo-rear-end.jpg')
    
    plate_input = plate_reader('Plate_examples/nick_test.jpg')
    print(plate_input)
    state_input = 'MN'
    url = "https://us-license-plate-to-vin.p.rapidapi.com/licenseplate"
    querystring = {"plate":plate_input,"state":state_input}
    headers = {
        'x-rapidapi-host': "us-license-plate-to-vin.p.rapidapi.com",
        'x-rapidapi-key': "60502a255cmsh1ceceb99f5c79a3p1963a9jsn231eb1b4b9a9"
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    plate_data_df = pd.DataFrame(response.json())
    plate_data_dict = plate_data_df.to_dict()
    make = plate_data_dict['specifications']['make']
    model = plate_data_dict['specifications']['model']
    year = plate_data_dict['specifications']['year']
    vin = plate_data_dict['specifications']['vin']
    num = plate_data_dict['plate']['make']
    state = plate_data_dict['state']['make']
    img_url = 'https://raw.githubusercontent.com/N-Prentko/ML_project_license_plate_identifier/master/BenReader/Plate_examples/nick_test.jpg'

    # Return template and data
    return render_template("index.html", img_url=img_url, num=num, state=state, make=make, model=model, year=year, vin=vin)

if __name__ == "__main__":
    app.run(debug=True)
