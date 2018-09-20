import cv2
import os
import numpy as np
import pickle
import sys


class FaceRecognition:
    def __init__(self):
        self.counter = 1
        self.id_counts = -1

        self.CURRENT_PATH = None
        self.NEW_PATH = None
        self.Face_Cascade = None

        self.__is_folder_existed = False
        self.__is_classifier_existed = False
        self.is_recording_enabled = False
        self.__is_model_existed = False

        self.image_buffer = []
        self.lables_buffer = []
        self.croped_image = np.array([])
        self.available_extentions = ["png" , 'pgm','jpeg', 'jpg', 'jpe']

        self.users_dict= {}

    def create_data_set_folder(self, folder_name=None):
        if not folder_name is None:
            self.CURRENT_PATH = os.getcwd()
            self.NEW_PATH = os.path.join(self.CURRENT_PATH, folder_name)
            self.__is_folder_existed = True

            if not os.path.exists(self.NEW_PATH):
                os.makedirs(self.NEW_PATH)
                self.__is_folder_existed = True
            print("the folder of the data set has been created")

        else:
            raise ValueError('you must put the data set folder name')

        return self.NEW_PATH

    def classifier(self, classifier_type = None):
        if not classifier_type is None:
            classifier_path = str(self.CURRENT_PATH + '/' + classifier_type)
            print(classifier_path)
            self.Face_Cascade = cv2.CascadeClassifier(classifier_path)
            self.__is_classifier_existed = True
            print('the classifier type is :', classifier_type)
        else:
            raise ValueError('you must put the classifier type ')

        return classifier_path

    def taking_traing_pic(self, number_of_pic=0, kill_button='s'):
        if self.__is_classifier_existed is True:
            kill = str(kill_button)
            camera = cv2.VideoCapture(0)
            while True:
                ret, camera_image = camera.read()
                gray_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
                faces = self.Face_Cascade.detectMultiScale(gray_image)

                for x, y, w, h in faces:
                    cv2.rectangle(camera_image, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)
                    cv2.putText(camera_image, 'press g to start ', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    self.cropped_image = np.array(camera_image[y:y + h, x:x + w])

                if ret == True and self.is_recording_enabled:
                    token_image_path = os.path.join(self.NEW_PATH, 'img_{}.png'.format(self.counter))
                    cv2.imwrite(token_image_path, self.cropped_image)
                    self.counter += 1
                    cv2.waitKey(50)

                    if self.counter >= number_of_pic:
                        print("the {} pictures were token".format(number_of_pic))
                        self.is_recording_enabled = False

                cv2.imshow('Mugiwara',camera_image)

                key = cv2.waitKey(50) & 255
                if key == ord(kill):
                    break
                elif key == ord('g'):
                    self.is_recording_enabled = True

                # destroy all opening windows after loop is finished
            cv2.destroyAllWindows()

        else:
            raise ValueError('you must put the classifier type first')

    def read_users_data_sets(self, users_folder_path):
        users_folder_names = os.listdir(users_folder_path)

        for every_user_name in users_folder_names:
            user_folder_path = os.path.join(users_folder_path,every_user_name)
            user_images = os.listdir(user_folder_path)
            self.id_counts +=1
            self.users_dict[every_user_name] = self.id_counts

            # path kol sora f kol folder
            for every_image in user_images :
                image_path =  os.path.join(user_folder_path,every_image)
                images_extension = every_image.split('.')[-1].lower()

                if images_extension in self.available_extentions :
                    image_read = cv2.imread(image_path,0)

                    if not image_read is None :
                        self.image_buffer.append(image_read)
                        self.lables_buffer.append(self.id_counts)

        images_number= len(self.image_buffer)
        users_labels = len(self.lables_buffer)
        print(images_number, users_labels)

        labels_as_np = np.asarray(self.lables_buffer)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(self.image_buffer, labels_as_np)
        face_recognizer.save("indvidual_faces.xml")


        pickle.dump(self.users_dict, open("user_details.info", 'wb'))


    def start_face_recognision(self):
        face_calssifier = cv2.CascadeClassifier(self.classifier('lbpcascade_frontalface.xml'))
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read("indvidual_faces.xml")
        mohamed = pickle.load(open("user_details.info", 'rb'))
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("pls connect the cammera")
            sys.exit(1)
        while True:
            ret, image = cam.read()
            if ret:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces_result = face_calssifier.detectMultiScale(gray_image)
                for x, y, w, h in faces_result:
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    sub_image = gray_image[y:y + h, x:x + w]
                    resized_image = cv2.resize(sub_image, (168, 192))
                    result = face_recognizer.predict(resized_image)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (75, 0, 130), 5)
                    if result[0] == 1:
                        qaqa = 'omar'
                        print('omar')
                    else:
                        qaqa = 'yousif'
                        print("yousif")
                    # cv2.putText(image, qaqa, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 255, 155))
                    cv2.putText(image, qaqa, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (124, 255, 0), 2)
                cv2.imshow("hello", image)
                if cv2.waitKey(10) & 255 == ord('s'):
                    break

    # def run_the_face_detection(self):
    #     camera = cv2.VideoCapture(0)
    #     while True:
    #         ret, camera_image = camera.read()
    #         gray_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    #         faces = self.Face_Cascade.detectMultiScale(gray_image)
    #
    #         for x, y, w, h in faces:
    #             cv2.rectangle(camera_image, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)
    #
    #         cv2.imshow('Mugiwara', camera_image)
    #
    #         if cv2.waitKey(50) & 255 == ord('s'):
    #             break
    #
    #
    #
    #
    #     cv2.destroyAllWindows()

