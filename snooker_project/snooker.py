#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import os
import glob
import math
import sys

from collections import Counter


# In[ ]:





# In[2]:


COLOR_TEMPLATES_PATH = 'color_hist_matching'
TEMPLATES_PATH = 'template_matching'
# IMAGES_PATH = 'training_data/Task1'
# VIDEOS_PATH = 'training_data/Task2'
# VIDEOS_PATH_T3 = 'training_data/Task3'

THRESHOLDS = [0.8, 0.76, 0.75, 0.78, 0.82, 0.82, 
              0.8, 0.83, 0.75, 0.8, 0.75, 0.87, 
              0.82, 0.85, 0.85, 0.82, 0.85, 
              0.8, 0.8, 0.82, 0.78, 0.78, 0.75, 0.87, 0.85, 
              0.85]


# ### Import Functions

# In[3]:


def import_templates(path):
    templates = []
    color_dict = {}
    templates_names = glob.glob(os.path.join(path, "*.jpg")) 

    for idx, name in enumerate(templates_names):      
        templates.append(cv.imread(name)) 
        color_dict[idx] = name.split('/')[-1][:-4]
    
    return (templates, color_dict)


# In[4]:


def import_images(path):
    images = []
    images_names = []
    
    for name in os.listdir(path):
        if 'jpg' in name:
            images.append(cv.imread(os.path.join(path, name)))
            images_names.append(name)
            
    return (images, images_names)


# In[46]:


def import_labels(path, img_names, task3=False):
    labels = []
    if task3:
        for name in img_names:
            with open(os.path.join(path, name), 'r') as file:
                labels.append(file.read())
    else:
        for name in img_names:
            file_name =  name.replace('jpg', 'txt')[:-4] + '_gt' + name.replace('jpg', 'txt')[-4:]

            with open(os.path.join(path, file_name), 'r') as file:
                labels.append(file.read())
    return labels


# In[6]:


def import_video(path, index):
    video = []
    cap = cv.VideoCapture(os.path.join(path, '{}.mp4'.format(index)))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            video.append(frame)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()

    cv.destroyAllWindows()
    
    return video


# ### Auxiliary Functions

# In[7]:


def display_image(image):
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# In[8]:


def save_image(image, base, step, name, index):
    
    if not os.path.exists(os.path.join(base, step)):
        os.makedirs(os.path.join(base, step))
    try:
        cv.imwrite(os.path.join(base, step, name) + '_{}.png'.format(index), image)
    except:
        print(' ')


# In[9]:


def draw_squares(image, squares, save=False, display=False):
    
    gray_image  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    draw = np.stack([gray_image, gray_image, gray_image], axis=2)
    
    for square in squares:
        cv.rectangle(draw, square[1], square[0], (0,0,255), 1)
    
    if save:
        print('TO DO')
#         save_image(draw, 'partial_results/testing', 'testing', 'video_{}'.format(i), index)
    if display:
        display_image(draw)

    return draw


# In[10]:


def draw_lines(image, lines ,color=(0, 0, 255)):
    
    gray_image  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    draw = np.stack([gray_image, gray_image, gray_image], axis=2)
    
    for line in lines:
        cv.line(draw, line[0], line[1], color, 2)
    
    return draw


# ### Crop, TM, IOU, Color

# In[11]:


def iou_score(boxA, boxB):
#     from here: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    xa = max(boxA[0][1], boxB[0][1])
    ya = max(boxA[0][0], boxB[0][0])
    xb = min(boxA[1][1], boxB[1][1])
    yb = min(boxA[1][0], boxB[1][0])

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    
    area_a = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    area_b = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    
    iou = intersection / float(area_a + area_b - intersection)

    return iou


# In[12]:


def find_table(image, index):
    # from lab 5
    low_green = (46, 100, 0)
    high_green = (85, 255, 255)

    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_table_hsv = cv.inRange(frame_hsv, low_green, high_green)

    table = cv.bitwise_and(image, image, mask=mask_table_hsv) 
    # detect edges on the result
    edges = cv.Canny(table, 75,150, apertureSize = 3)
    # detect lines
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=150, maxLineGap=100)

    horizontal_lines = [[(line[0], line[1]), (line[2], line[3])] for line in np.squeeze(lines) if math.isclose(line[1], line[3], abs_tol = 10) ]
    vertical_lines = [[(line[0], line[1]), (line[2], line[3])] for line in np.squeeze(lines) if [(line[0], line[1]), (line[2], line[3])] not in horizontal_lines ]
    
    l = len(vertical_lines)
    
    arr = np.array(vertical_lines).reshape((l, 4))

    min0 = np.min(arr[:, 0])
    min1 = np.min(arr[:, 1])
    min2 = np.min(arr[:, 2])
    min3 = np.min(arr[:, 3])
    
    max0 = np.max(arr[:, 0])
    max1 = np.max(arr[:, 1])
    max2 = np.max(arr[:, 2])
    max3 = np.max(arr[:, 3])
    
    min_h = min(min1, min3)
    max_h = max(max1, max3)
    
    min_w = min(min0, min2)
    max_w = max(max0, max2)

    return image[ min_h:max_h, min_w:max_w]


# In[13]:


# run template matching using a threshold from lab 6
def match_templates(image, index, thresholds_list, templates, display=False, save=False):
    frame = image.copy() 
    idx = -1
    all_matches = []
    
    for template in templates:    
        idx = idx + 1
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)    
        w, h = template_gray.shape[::-1]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        res = cv.matchTemplate(frame_gray, template_gray, cv.TM_CCOEFF_NORMED)

        threshold = thresholds_list[idx]
        loc = np.where( res >= threshold)
        all_matches.append(loc)

    return all_matches


# In[14]:


def get_squares(match_temp, templates):
    centers = []
    squares = []

    for i, template in enumerate(templates):
        w, h = template.shape[:-1]
        for pt in zip(*match_temp[i][::-1]):
            squares.append([(pt[0], pt[1]), (pt[0] + h, pt[1] + w)])
            centers.append([pt[0]+w//2, pt[1]+h//2])

    return squares


# In[15]:


def get_final_squares(squares, idx, iou_value):
    l = len(squares)
    matrix = np.zeros(l * l).reshape((l, l))

    for idx1, s1 in enumerate(squares):
        for idx2, s2 in enumerate(squares):
            matrix[idx1][idx2] = iou_score(s1, s2)
            
    aux = []

    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if matrix[i][j] > iou_value:
                aux.append(j)
                
    return [squares[ix] for ix in range(l) if ix not in aux]
    


# In[16]:


def squares_to_image_parts(squares_list, tables, index):
    images_parts = []

    for sq in squares_list:
            images_parts.append(tables[index][sq[0][1]:sq[1][1], sq[0][0]:sq[1][0]])

    return images_parts


# In[17]:


def decode_color(name):
    if 'black' in name:
        return 'black'
    elif 'blue' in name:
        return 'blue'
    elif 'brown' in name:
        return 'brown'
    elif 'green' in name:
        return 'green'
    elif 'pink' in name:
        return 'pink'
    elif 'red' in name:
        return 'red'
    elif 'white' in name:
        return 'white'
    elif 'yellow' in name:
        return 'yellow'
    elif 'background' in name:
        return 'background'


# In[18]:


# from lab 6
def find_color(img, color_templates, color_dict2, index, method, color_space):

    hist_templates = []
    for template in color_templates:
        template_hist = cv.calcHist([template], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist_templates.append(cv.cvtColor(template_hist, eval(color_space)))
    
    hist_img = cv.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist_img = cv.cvtColor(hist_img, eval(color_space))
    distances = []
    for i in range(len(color_templates)):
        
        hist_img_norm = hist_img / (hist_img.sum())
        
        hist_template_norm = hist_templates[i] / (hist_templates[i].sum())    

        dist = cv.compareHist(hist_img_norm, hist_template_norm, eval(method))
        distances.append(dist)

    color = decode_color(color_dict2[np.argmin(distances)])

    return color

    


# In[19]:


def count_balls(squares_list, color_templates, color_dict, image_index):
    balls = {'white': 0, 'black': 0, 'pink': 0, 'blue': 0, 'green': 0, 'brown': 0 , 'yellow': 0, 'red': 0, 'background':0}
    aux = []
    for j, el in enumerate(squares_list):
        color = find_color(el, color_templates, color_dict, str(image_index)+'_'+ str(j), 'cv.HISTCMP_CHISQR_ALT', 'cv.COLOR_BGR2BGRA')
        if color == 'background':
            aux.append(j)
        balls[color] += 1

    squares_list = [squares_list[i] for i in range(0, len(squares_list)) if i not in aux]

    del balls['background']
    return balls


# In[20]:


def check_colors(balls_dict):
    list_colors = ['white', 'black', 'pink', 'blue', 'green', 'brown', 'yellow']

    for el in list_colors:
        if balls_dict[el] > 1:
            balls_dict[el] = 1
    if balls_dict['red'] > 15:
        balls_dict['red'] = 15
        
    return balls_dict


# ### Output Functions

# In[21]:


def output_results_task1(path, balls_dict, index):
    balls_dict = check_colors(balls_dict)
        
    if not os.path.exists(path):
        os.makedirs(path)
    
    s = sum(balls_dict.values())
    with open(path +'/'+ str(index)+'.txt', 'w') as file:
        file.write(str(s))
        file.write('\n')
    for key in balls_dict:
        with open(path+'/' + str(index)+'.txt', 'a') as file:
            file.write(str(balls_dict[key]) + ' ' + key + '\n')


# In[22]:


def output_results_task2(path, not_potted, colors):
        
    if not os.path.exists(path):
        os.makedirs(path)
    for index in not_potted:
        with open(path +'/'+ str(index)+'.txt', 'w') as file:
            file.write('NO')
    
    for el in colors:
        with open(path+'/' + str(el[0])+'.txt', 'w') as file:
            file.write('YES\n' + '0\n' + str(el[1]))


# In[26]:


def output_results_task3(path, squares_list, ball_nr, video_index):
        
    if not os.path.exists(path):
        os.makedirs(path)
        
    with open(path +'/'+ str(video_index)+'_ball_'+str(ball_nr+1)+'.txt', 'a') as file:
            file.write(str(len(squares_list)) + ' -1 -1 -1 -1\n')
            
    for index, square in enumerate(squares_list):
        with open(path +'/'+ str(video_index)+'_ball_'+str(ball_nr+1)+'.txt', 'a') as file:
            file.write(str(index) + ' ' +str(square[0][0])+ ' ' + str(square[0][1])+ ' ' +str(square[1][0])+ ' '+str(square[1][1])+ '\n')


# In[27]:


def check_labels_task1(color_dict, label, index, errors):
    color_dict = check_colors(color_dict)

    label = label.split('\n')

    if int(label[0])  != sum(color_dict.values()):
        print('Different number of balls, image: {}'.format(index))
        errors.append(index)
    for color in label[1:]:
        c = color.split(' ')
        if int(c[0]) != color_dict[c[1]]:
            errors.append(index)
            print('\nimage: {}\ncolor: {}\nmodel output: {}\ntrue label: {}'.format(index, c[1], color_dict[c[1]], c[0]))


# ### First Task

# In[28]:


def test_first_task(IMAGES_PATH, LABELS_PATH=None, partial_results=False, ground_truth=False):
    # templates used for balls detection
    templates, color_dict = import_templates(TEMPLATES_PATH)
    # templates used for color identification
    color_templates, color_dict2 = import_templates(COLOR_TEMPLATES_PATH)
    # import images
    images, images_names = import_images(IMAGES_PATH)
    # extract table
    tables = [find_table(img, index) for index, img in enumerate(images)]
    
    if partial_results == 'True':
        for index, tab in enumerate(tables):
            save_image(tab, 'partial_results/Task1', 'cropped_tables', 'image', index+1)
    # template matching 
    matched = [match_templates(img, idx, THRESHOLDS, templates) for idx, img in enumerate(tables)]
    all_squares = [get_squares(match, templates) for match in matched]
    
    if partial_results == 'True':
        for index, sq_list in enumerate(all_squares):
            save_image(draw_squares(tables[index], sq_list), 'partial_results/Task1', 'all_templates_matched', 'image', index+1)
    # get just one template for each ball
    final_squares = [get_final_squares(s, i, 0.3) for i, s in enumerate(all_squares)]
    
    if partial_results == 'True':
        for index, sq_list in enumerate(final_squares):
            save_image(draw_squares(tables[index], sq_list), 'partial_results/Task1', 'final_templates', 'image', index+1)
    # extract the ball from the image
    images_parts = [squares_to_image_parts(sq_list, tables, index) for index, sq_list in enumerate(final_squares)]
    # determine color
    color_balls = [count_balls(img_parts, color_templates, color_dict2, index) for index, img_parts in enumerate(images_parts)]
    # if we have labels
    if ground_truth == 'True':
        labels = import_labels(LABELS_PATH, images_names)
        errors = []
        for idx, label in enumerate(labels):
            check_labels_task1(color_balls[idx], label, idx, errors)
        print('Errors are made in images: ', set(errors))
        print('Accuracy: ', 50 - len(set(errors)))
    # output results
    for idx, cb in enumerate(color_balls):
        output_results_task1('results/task1', cb, idx+1)


# ### SECOND TASK

# In[29]:


def check_potted_ball(dict_list):
    for el in dict_list[0]:
        if dict_list[0][el] != dict_list[1][el]:
            return el
    return False


# In[30]:


def potted_ball(frames_list, video_index):
    tab = [find_table(img, idx) for idx, img in enumerate(frames_list)]

    templates, color_dict = import_templates(TEMPLATES_PATH)

    color_templates, color_dict2 = import_templates(COLOR_TEMPLATES_PATH)
    matched = [match_templates(img, idx, THRESHOLDS, templates) for idx, img in enumerate(tab)]

    all_squares = [get_squares(match, templates) for match in matched]

    final_squares = [get_final_squares(s, i, 0.3) for i, s in enumerate(all_squares)]

    images_parts = [squares_to_image_parts(sq_list, tab, index) for index, sq_list in enumerate(final_squares)]

    color_balls = [count_balls(img_parts, color_templates, color_dict2, index) for index, img_parts in enumerate(images_parts)]

    checked = [check_colors(d) for d in color_balls]

    potted_ball = check_potted_ball(color_balls)
    
    return potted_ball
    


# In[31]:


def detect_balls(frame,  frame_index):

    tab = find_table(frame, frame_index)

    templates, color_dict = import_templates(TEMPLATES_PATH)
    color_templates, color_dict2 = import_templates(COLOR_TEMPLATES_PATH)
    
    matched = match_templates(tab, frame_index, THRESHOLDS, templates)
    all_squares = get_squares(matched, templates)

    final_squares = get_final_squares(all_squares, frame_index, 0.3)

    images_parts = squares_to_image_parts(final_squares, [tab], 0)

    color_balls = count_balls(images_parts, color_templates, color_dict2, frame_index)

    checked = check_colors(color_balls)

#     print('frame index: ', frame_index, 'sum: ', sum(checked.values()))
    return sum(checked.values())


# In[32]:


def potted(i, init, fin, not_potted):
#     print('video: {} init: {} fin: {}'.format(i+1, init, fin))
    if init <= fin:
#         print('NO')
        not_potted.append(i+1)


# In[33]:


def test_second_task(VIDEOS_PATH):
    all_videos = []

    for video_number in range(1, 26):
        num_of_balls = []
        print('Detect balls video ', video_number)
        v = import_video(VIDEOS_PATH, video_number)
        for index in range(0, len(v)-5, 3):
            balls_per_frame = detect_balls(v[index], index)
            num_of_balls.append(balls_per_frame)
        all_videos.append(num_of_balls)
        
    not_potted = []
    for index, list_balls in enumerate(all_videos):
        potted(index,  Counter(list_balls[:10]).most_common()[0][0], Counter(list_balls[-10:]).most_common()[0][0], not_potted)
    
    colors = []
    for index in range(1, 26):
        if index not in not_potted:
            print('Detect color for potted ball video ', index)
            v = import_video(VIDEOS_PATH, index)
            print('video: {}'.format(index))
            colors.append([index, potted_ball([v[0], v[-10]], index)])
    output_results_task2('results/task2', not_potted, colors)


# ### TASK 3

# In[34]:


def get_initial_boxes(video_index, VIDEOS_PATH_T3):
    test = import_labels(VIDEOS_PATH_T3, ['{}_ball_1.txt'.format(video_index), '{}_ball_2.txt'.format(video_index)], task3=True)
    test = [np.array(test[idx].split('\n')[1].split(' ')).astype(np.int) for idx in [0, 1]]
    test = [[(test[idx][1], test[idx][2]), (test[idx][3], test[idx][4])] for idx in [0, 1]]
    
    return test


# In[52]:


def eval_tracking(LABELS_PATH, frame, squares_list, video_index, ball_nr):
    gt = import_labels(LABELS_PATH, ['{}_ball_{}.txt'.format(video_index, ball_nr+1)])
    ct = 0
    true_squares = []
    for element in gt[0].split('\n')[1:-1]:
        try:
            lst = element.split(' ')
            ba = [(int(lst[1]), int(lst[2])), (int(lst[3]), int(lst[4]))]
            bb = squares_list[int(lst[0])]
            true_squares.append(ba)
    #         print('frame ' + lst[0] + ': ', iou_score(ba, bb))
            if iou_score(ba, bb) >= 0.2:
                ct += 1
        except:
            break

    print('video {}, ball {}: percent of frames with iou >= 0.2: '.format(video_index, ball_nr+1), round(100*ct/len(true_squares), 4))
#     save_image(draw_squares(frame, true_squares), 'partial_results/Task3', 'tracking', 'gt_video{}_ball'.format(video_index), ball_nr+1)


# In[54]:


def test_fun(frame, var1, var2, var3, var4, template, win_width, win_height):
    template_hist = cv.calcHist([template], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    template_hist = cv.cvtColor(template_hist, cv.COLOR_BGR2BGRA)
    hist_template_norm = template_hist / (template_hist.sum()) 

    distances = []
    points_pairs = []
    
    for i in range(max(var2-win_width, 0), min(var4+win_width, frame.shape[0]-win_width)):
        for j in range(max(var1-win_height, 0), min(var3+win_height, frame.shape[1]-win_height)):
            hist_img = cv.calcHist(frame[i:i+win_width, j:j+win_height], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            hist_img = cv.cvtColor(hist_img, cv.COLOR_BGR2BGRA)

            hist_img_norm = hist_img / (hist_img.sum())

            dist = cv.compareHist(hist_img_norm, hist_template_norm, cv.HISTCMP_CHISQR_ALT)
            distances.append(dist)
            points_pairs.append([j, i])

    template = frame[ points_pairs[np.argmin(distances)][1]:points_pairs[np.argmin(distances)][1]+win_width, points_pairs[np.argmin(distances)][0]:points_pairs[np.argmin(distances)][0]+win_height]

    var1 = points_pairs[np.argmin(distances)][0]
    var2 = points_pairs[np.argmin(distances)][1]
    var3 = points_pairs[np.argmin(distances)][0] + win_width
    var4 = points_pairs[np.argmin(distances)][1] + win_height
    return (var1, var2, var3, var4, template)


# In[44]:


def repeated_detection(video_index, VIDEOS_PATH_T3, partial_results=False, ground_truth=False, LABELS_PATH=None):
    video = import_video(VIDEOS_PATH_T3, video_index)
    
    boxes = get_initial_boxes(video_index, VIDEOS_PATH_T3)

    custom_templates = [video[0][boxes[idx][0][1]: boxes[idx][1][1], boxes[idx][0][0]: boxes[idx][1][0]] for idx in [0, 1]]

    for ball_nr in [0, 1]:
        TEMPLATE = custom_templates[ball_nr].copy()
        win_width, win_height = custom_templates[ball_nr].shape[:-1]
        
        var1, var2, var3, var4 = boxes[ball_nr][0][0], boxes[ball_nr][0][1], boxes[ball_nr][1][0], boxes[ball_nr][1][1]
        squares_list = []

        for index in range(0, len(video)-5):
            var1, var2, var3, var4, TEMPLATE = test_fun(video[index], var1, var2, var3, var4, TEMPLATE, win_width, win_height)
            squares_list.append([(var1, var2), (var3, var4)])
            
        if partial_results == 'True': 
            save_image(draw_squares(video[0], squares_list), 'partial_results/Task3', 'tracking', 'video{}_ball'.format(video_index), ball_nr+1)
        if ground_truth == 'True':
            eval_tracking(LABELS_PATH, video[0], squares_list, video_index, ball_nr)
            
        output_results_task3('results/task3', squares_list, ball_nr, video_index)


# In[47]:


def test_third_task(VIDEOS_PATH, partial_results, ground_truth, LABELS_PATH=None):
    for index in range(1, 26):
        print('video: ', index)
        repeated_detection(index, VIDEOS_PATH, partial_results, ground_truth, LABELS_PATH)


# In[55]:


# test_third_task('training_data/Task3', 'True', 'True', 'training_data/Task3/ground-truth')


# In[ ]:


# from command line
TASK = sys.argv[1]

if TASK == 'task1':

    IMAGES_PATH = sys.argv[2]
    partial_results = sys.argv[3]
    ground_truth = sys.argv[4]
    
    if ground_truth:
        LABELS_PATH = sys.argv[5]
    else:
        LABELS_PATH = None
    
    test_first_task(IMAGES_PATH, LABELS_PATH, partial_results, ground_truth)

elif TASK == 'task2':
    VIDEOS_PATH = sys.argv[2]
    test_second_task(VIDEOS_PATH)

elif TASK == 'task3':
    VIDEOS_PATH = sys.argv[2]
    partial_results = sys.argv[3]
    ground_truth = sys.argv[4]
    
    if ground_truth == 'True':
        LABELS_PATH = sys.argv[5]
    else:
        LABELS_PATH = None
    
    test_third_task(VIDEOS_PATH, partial_results, ground_truth, LABELS_PATH)

