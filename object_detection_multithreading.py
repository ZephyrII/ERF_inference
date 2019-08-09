import os
import cv2
import time
import argparse
import numpy as np

np.set_printoptions(threshold=np.inf)
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from collections import OrderedDict
sum_det = 0

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = os.path.join("/root/Faster/model/31.01/SBC50", 'erf_cityscapes.pb')
PATH_TO_CKPT = os.path.join("/root/share/tf/", 'erf_ouster.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("/root/share/tf/dataset", 'label_map.pbtxt')

NUM_CLASSES = 1


# Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
#                                                             use_display_name=True)
# category_index = label_map_util.create_category_index(categories)


def normalize_map(f_map):
    m_min = f_map.min()
    m_max = f_map.max()
    f_map = ((f_map - m_min) * 255.0 / (m_max - m_min))
    return f_map


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('input_data:0')

    # Each box represents a part of the image where a particular object was detected.
    output_data = detection_graph.get_tensor_by_name('output_data:0')
    output_classes = detection_graph.get_tensor_by_name('output_classes:0')
    output_colored = detection_graph.get_tensor_by_name('output_colored:0')
    # Actual detection.
    (output_data, output_classes, output_colored) = sess.run([output_data, output_classes, output_colored],
                                                             feed_dict={image_tensor: image_np_expanded})

    return dict(output_data=output_data, output_classes=output_classes, output_colored=output_colored)


def worker(frame): #input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # fps = FPS().start()
    # while True:
    #     fps.update()
        # frame = input_q.get()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data = detect_objects(frame_rgb, sess, detection_graph)
    # data = output_q.get()
    output_data = np.squeeze(data['output_data'])
    output_color = np.squeeze(data['output_colored'])
    output_classes = data['output_classes']
    # print("uni", np.unique(output_data))
    out = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
    # fps.stop()
    sess.close()
    return out


def display_frame(frame):
    # category_mappings = {"flat": ([0, 1], [20, 20, 20]),
    #                      "construction": ([2, 3, 4], [100, 100, 100]),
    #                      "object": ([5, 6, 7], [225, 225, 0]),
    #                      "vegetation": ([8], [0, 200, 0]),
    #                      "terrain": ([9], [0, 80, 0]),
    #                      "sky": ([10], [255, 255, 255]),
    #                      "human": ([11, 12], [255, 0, 0]),
    #                      "vehicle": ([13, 14, 15, 16, 17, 18], [0, 0, 255])}
    category_mappings = [[0, 1], [2, 3, 4], [5, 6, 7], [8], [9],[10], [11, 12], [13, 14, 15, 16, 17, 18]]
    color_mappings = [[20, 20, 20], [100, 100, 100], [225, 225, 0], [0, 220, 0], [0, 80, 0], [255, 255, 255], [255, 0, 0], [0, 0, 150]]

    if output_q.empty():
        pass  # fill up queue
    else:
        data = output_q.get()
        output_data = np.squeeze(data['output_data'])
        output_color = np.squeeze(data['output_colored'])
        output_classes = data['output_classes']
        print("uni", np.unique(output_data))

        # out_img = np.zeros((720, 1280, 3))
        # category_images = []
        # for value in category_mappings:
        #     images_to_merge =[]
        #     for id in value:
        #         images_to_merge.append(output_data[:, :, id])
        #     category_images.append(np.max(images_to_merge, axis=0))
        # category_images = np.stack(category_images)
        # max_id = np.argmax(category_images, axis=0)
        #
        # for idx, value in enumerate(color_mappings):
        #     out_img[max_id == idx] = value
        # out_img = cv2.resize(out_img, None, fx=0.9, fy=0.9)
        # out_img = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        out = cv2.cvtColor(output_color, cv2.COLOR_BGR2RGB)
        cv2.imshow('Videfo', out)
        return out
        # output_colored = data['output_colored'][0]
        # output_colored = cv2.cvtColor(output_colored, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Video', output_colored / 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/root/share/tf/object_detector_app/5.avi')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    args = parser.parse_args()



    fps = FPS().start()
    if args.image_directory:
        print('Reading from image folder')
        images_mode = True
        images = iter(os.listdir(args.image_directory))
        cv2.namedWindow("Video")
        frame = None
        name = None
        while True:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            if True: #k == ord('n'):
                try:
                    name = next(images)
                    if name == 'seg':
                        continue
                    fname = os.path.join(args.image_directory, name)
                    print(fname)
                    frame = cv2.imread(fname)
                    # frame = cv2.resize(frame, (1280, 720))
                    # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    # clahe = cv2.createCLAHE(clipLimit=2.0,  tileGridSize=(8, 8))
                    # img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                    # frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    cv2.imshow('Videggo', frame)
                    if frame is not None:
                        # input_q.put(frame)
                        out_fr = worker(frame)
                    out_fname = os.path.join(args.image_directory, "seg", name)
                    cv2.imwrite(out_fname, out_fr)
                except StopIteration:
                    break
    else:
        print('Reading from file.', args.video_input)
        video_capture = cv2.VideoCapture(args.video_input)
        ret = True
        frame = None
        while ret:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            t = time.time()
            ret, frame = video_capture.read()
            if frame is not None:
                img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
                frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

                frame = cv2.resize(frame, (1280, 720))
                input_q.put(frame)
                # frame = cv2.resize(frame, None, fx=0.9, fy=0.9)
                cv2.imshow('Videggo', frame)
                display_frame(frame)

            fps.update()
        video_capture.stop()
    #        print('[INFO] elapsed time: {:.4f}'.format(time.time() - t))

    fps.stop()
    print("Final sum det:", sum_det)
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cv2.destroyAllWindows()
