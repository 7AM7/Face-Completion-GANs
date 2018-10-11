from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(filename,size=(256, 256)):
    img = cv2.imread(filename)
    # resize
    img = cv2.resize(img, size)
    # 'RGB'->'BGR'
    img = img[:, :, ::-1]
    img = np.array(img, dtype=np.float32) / 255.0
    return img

def show_image(batch_size,n,inputs,masks,completion_image):
    r, c = 3, 3
    fig, axs = plt.subplots(r, c)
    for i in range(r):
        axs[i, 0].imshow(inputs[i] * (1 - masks[i]))
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input')
        axs[i, 1].imshow(completion_image[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Output')
        axs[i, 2].imshow(inputs[i])
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Ground Truth')
    fig.savefig("result/result_%d.png" % n)
    plt.show()

def get_points(batch_size, hole_min=64, hole_max=128):
    points = []
    mask = []
    for i in range(batch_size):
        x1, y1 = np.random.randint(0, 256 - 128 + 1, 2)
        x2, y2 = np.array([x1, y1]) + 128
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(hole_min, hole_max, 2)
        p1 = x1 + np.random.randint(0, 128 - w)
        q1 = y1 + np.random.randint(0, 128 - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((256, 256, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)

def plot_image(_img, _label, _num):
    plt.subplot(1, 3, _num)
    plt.imshow(_img)
    # plt.axis('off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.xlabel(_label)