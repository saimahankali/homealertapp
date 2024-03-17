from unicodedata import name
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image
import streamlit as st
import numpy as np
import base64
import warnings
warnings.filterwarnings('ignore')
import pickle,time
from keras.utils import img_to_array, load_img
from preprocessing import *
from create_embedding import get_embedding,convert_embedding
from facenet_architecture import InceptionResNetV2
from load import load_dataset
from extract_img_from_video import extract_frame
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from train import encoder
from sendmail import send_email

# global name_list,existing_names,train_X, train_y, test_X, test_y, facenet
facenet = InceptionResNetV2()
base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MTCNN')
path = os.path.join(base_dir,"model/Facenet_keras_weights.h5")
facenet.load_weights(path)
file_path = os.path.join(base_dir,'name_list.txt')
# Create name_list.txt file if it doesn't exist
if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        pass  # Just create an empty file
try:
    # Load existing names from the file
    with open(file_path, "r") as f:
        existing_names = [line for line in f if line.strip()]
    # Extract existing names (if any)
    name_list = [name.strip() for name in existing_names]

    data = np.load(os.path.join(base_dir,"model/Faces-dataset.npz"),allow_pickle=True)
    train_X, train_y, test_X, test_y = data['a'], data['b'], data['c'], data['d']

    data_embed  = np.load(os.path.join(root_url,"model/Face-dataset-embedding.npz"))

    X_train , y_train , X_test, y_test = data_embed["a"], data_embed["b"], data_embed["c"], data_embed["d"]

    file_name =os.path.join(base_dir,"model/classify.sav")
    loaded_model = pickle.load(open(file_name, "rb"))

    
except Exception as e:
    print(e)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def process_input(filename, target_size=(160, 160)):
    img = load_img(filename)
    img_arr = np.array(img)
    result = detector.detect_faces(img_arr)
    if len(result) == 0:
        return None, None
    else:
        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_arr[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(target_size)
        resized_arr = np.asarray(image)
    return image, resized_arr


def embed_input(model, resized_arr):
    embed_vector = get_embedding(facenet, resized_arr)
    return embed_vector


def predict(model, embed_vector):
    sample = np.expand_dims(embed_vector, axis=0)
    yhat_index = model.predict(sample)
    yhat_prob = np.max(model.predict_proba(sample)[0])
    class_predict = name_list[yhat_index[0]]
    return yhat_prob, class_predict

def capture_video(name, duration=15):
    save_path = os.path.join(base_dir,'Video')
    # Create the save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{name}.mp4")

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(save_file, fourcc, 20.0, (640, 480))

    start_time = cv2.getTickCount()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_count = int(duration * fps)  # Calculate total frames for the duration

    st.write("Capturing video. Please wait...")

    frame_placeholder = st.empty()  # Create an empty placeholder to display frames

    for _ in range(frame_count):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_placeholder.image(frame, channels="BGR")  # Display the frame
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time > duration:
                break
        else:
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.write(f"Video saved as {save_file}")


def update_name_list(name):
    all_names = list(set(existing_names + [name]))
    with open(file_path, "w") as f:
        for name in all_names:
            f.write(name + "\n")


def capture_15sec_video():
    st.subheader("Capture 15 sec video")
    name = st.text_input("Enter name for the video:")
    if st.button("Start Capture"):
        capture_video(name)
        # Update name list and save to file
        update_name_list(name)
        st.write("Video captured and saved successfully.")

def capture_video_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        # Convert the frame to PIL Image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
    else:
        frame_image = None
    cap.release()
    return frame_image

def save_video(frames, path, fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame in frames:
        out.write(frame)
    out.release()

def live_detect():
    st.title("Live Video Feed Recognition")
    start_button = st.button("Start Recognition")
    stop_button = st.button("Stop Recognition")

    unknown_images_dir = os.path.join(base_dir, "unknown_images")
    # Create the unknown_images directory if it doesn't exist
    os.makedirs(unknown_images_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    last_image_saved_time = time.time()
    last_email_sent_time = time.time()
    current_time = time.time()
    while start_button and not stop_button:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)

            result = detector.detect_faces(np.array(frame))
            recognition_result = None

            for face in result:
                x, y, width, height = face['box']
                face_array = frame_rgb[y:y+height, x:x+width]
                face_image = Image.fromarray(face_array).resize((160, 160))
                embed_vector = get_embedding(facenet, np.array(face_image))
                prob, name = predict(loaded_model, embed_vector)
                prob = np.round(prob, 4)
                if prob >= 1.0:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, f'{name}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    recognition_result = (prob, name)
                else:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    recognition_result = (prob, "Unknown")

                    # Save the face image every 5 minutes
                    current_time = time.time()
                    if current_time - last_image_saved_time >= 1 * 60:  # 5 minutes in seconds
                        face_image_path = os.path.join(unknown_images_dir, f"unknown_{time.strftime('%Y%m%d-%H%M%S')}.jpg")
                        face_image.save(face_image_path)
                        last_image_saved_time = current_time
            current_time = time.time()
            # Check if 30 minutes have passed since the last email
            if current_time - last_email_sent_time >= 1 * 60:  # 30 minutes in seconds
                try:
                    st.write(f'start:{face_image_path}')
                    # Send an email with the latest unknown face image
                    send_email(face_image_path)
                    st.write(f'end:{face_image_path}')
                    # Display a success message
                    st.success("Email sent with the unknown face image.")

                    # Update the time tracking variable
                    last_email_sent_time = current_time

                    # Wait for 10 seconds after sending the email
                    time.sleep(10)
                except Exception as e:
                    # Display an error message
                    st.error(f"Failed to send email: {e}")

            frame_placeholder.image(frame)

            if recognition_result:
                prob, name = recognition_result
                if name != "Unknown":
                    st.success(f'Recognised person is: {name} (Confidence: {prob:.2f})')
                else:
                    st.error('Recognised person is Unknown')

        if stop_button:
            break

    cap.release()

def live_video():
    st.title("Live Video Feed Recognition")
    start_button = st.button("Start Recognition")
    stop_button = st.button("Stop Recognition")

    unknown_videos_dir = os.path.join(base_dir, "unknown_videos")
    # Create the unknown_videos directory if it doesn't exist
    os.makedirs(unknown_videos_dir,exist_ok=True)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    unknown_start_time = None
    unknown_video_frames = []
    unknown_detected = False
    unknown_video_saved = False

    frame_rate = 3  # frames per second
    video_duration = 10  # seconds
    max_frames = frame_rate * video_duration
    # Initialize the time tracking variable
    last_email_sent_time = time.time()
    while start_button and not stop_button:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)

            result = detector.detect_faces(np.array(frame))
            recognition_result = None

            for face in result:
                x, y, width, height = face['box']
                face_array = frame_rgb[y:y+height, x:x+width]
                face_image = Image.fromarray(face_array).resize((160, 160))
                embed_vector = get_embedding(facenet, np.array(face_image))
                prob, name = predict(loaded_model, embed_vector)
                # prob = np.round(prob, 4)
                if prob >= 1.0:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, f'{name}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    recognition_result = (prob, name)
                    unknown_detected = False
                else:
                    # cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    # cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    recognition_result = (prob, "Unknown")
                    if not unknown_detected:
                        unknown_detected = True
                        unknown_start_time = time.time()
                        unknown_video_frames = []
                    unknown_video_frames.append(frame)

            if unknown_detected and len(unknown_video_frames) >= max_frames:
                unknown_detected = False
                unknown_video_saved = True
                video_path = os.path.join(unknown_videos_dir, f"unknown_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                save_video(unknown_video_frames, video_path, fps=frame_rate)


            if unknown_video_saved:
                # Update the flag
                unknown_video_saved = False
                
                # Check if 30 minutes have passed since the last email
                current_time = time.time()
                if current_time - last_email_sent_time >= 1 * 60:  # 30 minutes in seconds
                    try:
                        # Send an email with the latest unknown video
                        send_email(video_path)
                        
                        # Display a success message
                        st.success("Email sent with the latest unknown video.")
                        
                        # Update the time tracking variable
                        last_email_sent_time = current_time
                        time.sleep(20)
                    except Exception as e:
                        # Display an error message
                        st.error(f"Failed to send email: {e}")
            frame_placeholder.image(frame)

            if recognition_result:
                prob, name = recognition_result
                if name != "Unknown":
                    st.success(f'Recognised person is: {name} (Confidence: {prob:.2f})')
                else:
                    st.error('Recognised person is Unknown')

        if stop_button:
            break

    cap.release()

def get_known_count():
    known_videos_dir = os.path.join(base_dir, "Video")
    known_videos = [f for f in os.listdir(known_videos_dir) if os.path.isfile(os.path.join(known_videos_dir, f))]
    return len(known_videos)

def update_name_list(new_name):
    # Append the new name to the in-memory list
    if new_name not in name_list:
        name_list.append(new_name)

    # Write the updated list to the file
    with open(file_path, "w") as f:
        for name in name_list:
            f.write(f"{name}\n")

def main():
    global train_X, train_y, test_X, test_y,X_train , y_train , X_test, y_test
    # print(name_list)
    st.title("Face Recognition with MTCNN and FaceNet")
    options = ['Capture 15sec Video','Extract Image from Video', 'Preprocess', 'Load', 'Create Embeddings', 'Train', 'Recognise Face image', 'Live Detect']
    selected_option = st.selectbox("Choose an option", options)

    if selected_option == 'Capture 15sec Video':
        st.write('Video Capturing...... ')
        capture_15sec_video()

    elif selected_option=='Extract Image from Video':
        st.subheader("Extract images from video")
        if st.button("Preprocess"):
            st.write('Extraction in progress...')
            extract_frame(name_list)
            st.write('Extraction Completed!')

    elif selected_option == 'Preprocess':
        st.subheader("Preprocessing captured video")
        if st.button("Preprocess"):
            st.write('Preprocessing in progress...')
            extract_face_fromdir(name_list)
            train_test_seperate(name_list)
            st.write('Preprocess completed')

    elif selected_option == 'Load':
        st.subheader("Load and Split Dataset")
        if st.button("Load Dataset"):
            st.write('Loading in progress...')
            # Change link if we use new dataset
            train_X, train_y = load_dataset(os.path.join(base_dir,'Train'))
            st.write(train_X.shape, train_y.shape)
            # Similarly with new testset
            test_X, test_y = load_dataset(os.path.join(base_dir,'Test'))
            st.write(test_X.shape, test_y.shape)
            # save arrays to one file in compressed format
            np.savez_compressed(f"{os.path.join(base_dir,'model/Faces-dataset.npz')}", a = train_X, b = train_y, c = test_X, d = test_y)
            st.write("Train_X shape: {}".format(train_X.shape))
            st.write("Train_y shape: {}".format(train_y.shape))
            st.write("Test_X shape: {}".format(test_X.shape))
            st.write("Test_y shape: {}".format(test_y.shape))
            st.write('Load completed')



    elif selected_option == 'Create Embeddings':
        st.subheader("Extract Face Features")
        if st.button("Extrace Face"):
            st.write('Extracting in progress...')
            # Change link if we use new dataset
            embed_trainX, embed_testX = convert_embedding(train_X,test_X,facenet)
            np.savez_compressed(f"{os.path.join(base_dir,'model/face-dataset-embedding.npz')}", a = embed_trainX, b = train_y, c = embed_testX, d = test_y)
            st.write("Saved Embedding....")
            st.write("Embed train X: {}".format(embed_testX.shape))
            st.write("Train_y shape: {}".format(train_y.shape))
            st.write('Extract completed')


    elif selected_option == 'Train':
        st.subheader("Train Face Data")
        if st.button("Train Face"):
            st.write('Training in progress...')

            X_train_encode, y_train_encode, X_test_encode, y_test_encode = encoder(X_train , y_train , X_test, y_test)
            #SVM classifier
            model = SVC(kernel='linear', probability= True, random_state = 42)
            model.fit(X_train_encode, y_train_encode)

            yhat_train = model.predict(X_train_encode)
            yhat_test = model.predict(X_test_encode)

            # print(yhat_train)
            # print(yhat_test)

            score_train = accuracy_score(y_train_encode, yhat_train)
            score_test = accuracy_score(y_test_encode, yhat_test)

            st.write("Accuracy on train set: {}".format(score_train))
            st.write("Accuracy on test set: {}".format(score_test))
            #Save model
            filename = f"{os.path.join(base_dir,'model/classify.sav')}"
            pickle.dump(model, open(filename, 'wb'))

    elif selected_option == 'Recognise Face image':
        uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
        if uploaded_file is not None:
            st.write("File uploaded:", uploaded_file.name)
            show_img = load_img(uploaded_file, target_size=(300, 300))
            st.image(show_img, caption="Original image uploaded")
            save_dir = "MTCNN/image_from_user"
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            saveimg_dir = "MTCNN/image_from_user" + "\{}".format(uploaded_file.name)
            image, resized_arr = process_input(saveimg_dir)

            if (image == None and resized_arr == None):
                print("Can't detect face! Please try another image")
            else:
                embed_vector = embed_input(loaded_model, resized_arr)
                prob, pred_class = predict(loaded_model, embed_vector)
                prob = np.round(prob, 4)
                st.success('Predicted with confidence: {}'.format(np.round(prob, 4)))
                print(prob)

                if prob >= 1.0:
                    st.success(f'Recognised person is: {pred_class}')
                else:
                    st.error('Recognised person is Unknown')

    # For Image sending to mail use below Elif Statement
    # elif selected_option == 'Live Detect':
    #         live_detect()

    # For Video sending to mail use below Elif Statement
    elif selected_option == 'Live Detect':
            live_video()

    # Show unknown videos at the start of login
    unknown_videos_dir = os.path.join(base_dir, "unknown_videos")
    unknown_videos = [f for f in os.listdir(unknown_videos_dir) if os.path.isfile(os.path.join(unknown_videos_dir, f))]

    if unknown_videos:
        st.subheader("Unknown Videos")
        known_count = get_known_count()  # Get the current count of known videos

        # Create a grid layout for videos, 4 per row
        columns = st.columns(4)
        column_index = 0

        for i, video in enumerate(unknown_videos):
            # Display video in the next column
            with columns[column_index]:
                st.video(os.path.join(unknown_videos_dir, video))
                if st.button(f"Mark as Known {video}", key=video):
                    known_count += 1
                    new_name = f"known_{known_count}"
                    new_video_name = f"{new_name}.mp4"
                    known_video_path = os.path.join(base_dir, "Video", new_video_name)
                    
                    # Move and rename the video
                    os.rename(os.path.join(unknown_videos_dir, video), known_video_path)
                    update_name_list(new_name)  # Update name_list.txt with the new name
                    
                    st.success(f"Moved {video} to known videos as {new_video_name}")
                    st.rerun()  # Rerun the script to refresh the dashboard

            # Move to the next column, and wrap back to the first column if necessary
            column_index = (column_index + 1) % 4
    
if __name__ == '__main__':
    main()
