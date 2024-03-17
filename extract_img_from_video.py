import cv2
import os
from tqdm import tqdm

base_url = os.path.join(os.path.abspath(os.path.dirname(__file__)), "MTCNN")  # Change root_url according to computer directory

def extract_frame(name_list):
    root_url = os.path.join(base_url, "Video")
    img_folder = os.path.join(base_url, "Raw_images")
    for name in tqdm(name_list):
        video = os.path.join(root_url, name + ".mp4")
        vidcap = cv2.VideoCapture(video)
        saved_url = os.path.join(img_folder, name)
        os.makedirs(saved_url, exist_ok=True)
        count = 50
        leaf = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break
            if leaf % 3 == 0:
                cv2.imwrite(os.path.join(saved_url, "{}{}.png".format(name, count)), image)
                count -= 1
            leaf += 1
            if count < 0:
                break
        vidcap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     extract_frame(name_list)
