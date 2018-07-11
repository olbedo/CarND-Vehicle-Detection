# coding: utf-8

# # Vehicle Detection and Tracking Project
import os.path
import glob
import pickle
import time
from collections import deque
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Tweak these parameters and see how the results change.
COLOR_CONV = 'RGB2YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ADAPT_HIST = False
EQUALIZE = False
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
HIST_RANGE = (0, 256)
ORIENTATION = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
BLOCK_NORM = "L2-Hys"
HOG_CHANNEL = "ALL" # Can be 0, 1, 2, or "ALL"
SPATIAL_FEAT = True # Spatial features on or off
HIST_FEAT = True # Histogram features on or off
HOG_FEAT = True # HOG features on or off
APPENDIX = "_{}{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}.p".format(
        COLOR_CONV[4:] if COLOR_CONV else 'RGB', 
        '_AH' if ADAPT_HIST else '_EQ' if EQUALIZE else '',
        SPATIAL_SIZE[0], HIST_BINS, HIST_RANGE[1], ORIENTATION, 
        PIX_PER_CELL, CELL_PER_BLOCK, BLOCK_NORM, HOG_CHANNEL, 
        int(SPATIAL_FEAT), int(HIST_FEAT), int(HOG_FEAT))
CLASSIFIER = "LinSVC_CV"
SEARCH_PARAMS = [(350, 510, 0.75, 2), (380, 508, 1.0, 2), (395, 651, 1.5, 2), #(390, 656, 1.75, 2), 
                 (410, 666, 2.0, 2)] 
THRESHOLD = 70
LAST_N = 10
recent_boxes = deque(maxlen=LAST_N)


# Define a function to return some characteristics of the dataset 
def get_data_info(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Define a key "max_range" and store the maximum value of the color range of the test image.
    data_dict["max_range"] = 1 if img.max() <= 1 else 255
    # Return data_dict
    return data_dict

def convert_color(img, conv='RGB2YCrCb'):
    #img = (equalize_hist(img) * 255).astype(np.uint8)
    if conv is None:
        return img
    if conv == 'RGB2GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'HLS2RGB':
        return cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    if conv == 'HSV2RGB':
        return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    if conv == 'Lab2RGB':
        return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    if conv == 'LUV2RGB':
        return cv2.cvtColor(img, cv2.COLOR_LUV2RGB)
    if conv == 'YCrCb2RGB':
        return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    if conv == 'YUV2RGB':
        return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    raise ValueError("Unknown conversion format: {}".format(conv))

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size, interpolation=cv2.INTER_AREA).ravel()
    color2 = cv2.resize(img[:,:,1], size, interpolation=cv2.INTER_AREA).ravel()
    color3 = cv2.resize(img[:,:,2], size, interpolation=cv2.INTER_AREA).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32, bins_range=(0, 256), return_bin_centers=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    if not return_bin_centers:
        return hist_features
    else:
        # Return the individual histograms, bin_centers and feature vector
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        return hist_features, bin_centers

def get_hog_features(img, orient, pix_per_cell, cell_per_block, block_norm="L2-Hys", 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  block_norm=block_norm, transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm=block_norm, transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, mul255=True, color_conv=None, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, 
                     cell_per_block=2, block_norm="L2-Hys", hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img_file in imgs:
        # Read in each one by one
        img = mpimg.imread(img_file)
        # apply color conversions if requested
        if mul255:
            img = (img*255).astype(np.uint8)
        if color_conv:
            img = convert_color(img, color_conv)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(img, size=spatial_size)
        assert not np.isnan(np.sum(spatial_features))
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
        assert not np.isnan(np.sum(hist_features))
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:,:,channel], orient, 
                            pix_per_cell, cell_per_block, block_norm=block_norm, 
                            vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, block_norm=block_norm, 
                        vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        assert not np.isnan(np.sum(hog_features))
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

def train():
    # Read in car and non-car images
    car_files = glob.glob('training_images/vehicles/*/*.png')
    notcar_files = glob.glob('training_images/non-vehicles/*/*.png')
    data_info = get_data_info(car_files, notcar_files)
    print(data_info["n_cars"], "vehicle images found")
    print(data_info["n_notcars"], "non-vehicle images found")
    print("Image shape:", data_info["image_shape"])
    print("Data type:", data_info["data_type"])
    print("Maximum value of the color range:", data_info["max_range"])
    feat_pickle = "Feat" + APPENDIX

    if not os.path.exists(feat_pickle):
        t=time.time()
        car_features = extract_features(car_files, mul255=data_info["max_range"]==1, 
                    color_conv=COLOR_CONV, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                    hist_range=HIST_RANGE, orient=ORIENTATION, pix_per_cell=PIX_PER_CELL, 
                    cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL, block_norm=BLOCK_NORM, 
                    spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        notcar_features = extract_features(notcar_files, mul255=data_info["max_range"]==1, 
                    color_conv=COLOR_CONV, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                    hist_range=HIST_RANGE, orient=ORIENTATION, pix_per_cell=PIX_PER_CELL, 
                    cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL, block_norm=BLOCK_NORM, 
                    spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract features...')

        dist_pickle = {}
        dist_pickle["car_features"] = car_features
        dist_pickle["notcar_features"] = notcar_features
        dist_pickle["color_conv"] = COLOR_CONV
        dist_pickle["spatial_size"] = SPATIAL_SIZE
        dist_pickle["hist_bins"] = HIST_BINS
        dist_pickle["hist_range"] = HIST_RANGE
        dist_pickle["orient"] = ORIENTATION
        dist_pickle["pix_per_cell"] = PIX_PER_CELL
        dist_pickle["cell_per_block"] = CELL_PER_BLOCK
        dist_pickle["block_norm"] = BLOCK_NORM
        dist_pickle["hog_channel"] = HOG_CHANNEL
        dist_pickle["spatial_feat"] = SPATIAL_FEAT
        dist_pickle["hist_feat"] = HIST_FEAT
        dist_pickle["hog_feat"] = HOG_FEAT
        pickle.dump(dist_pickle, open(feat_pickle, "wb" ) )
        print('Features saved to pickle...') 
    else:
        dist_pickle = pickle.load( open(feat_pickle, "rb" ) )
        car_features = dist_pickle["car_features"]
        notcar_features = dist_pickle["notcar_features"]
        COLOR_CONV = dist_pickle["color_conv"]
        SPATIAL_SIZE = dist_pickle["spatial_size"]
        HIST_BINS = dist_pickle["hist_bins"]
        HIST_RANGE = dist_pickle["hist_range"]
        ORIENTATION = dist_pickle["orient"]
        PIX_PER_CELL = dist_pickle["pix_per_cell"]
        CELL_PER_BLOCK = dist_pickle["cell_per_block"]
        BLOCK_NORM = dist_pickle["block_norm"]
        HOG_CHANNEL = dist_pickle["hog_channel"]
        SPATIAL_FEAT = dist_pickle["spatial_feat"]
        HIST_FEAT = dist_pickle["hist_feat"]
        HOG_FEAT = dist_pickle["hog_feat"]
        print('Features loaded from pickle...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    if CLASSIFIER == "SVC":
        clf = SVC()
    elif CLASSIFIER == "AdaBoost":
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                 algorithm="SAMME.R", learning_rate=0.5)
    elif CLASSIFIER == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        # Use a linear SVC 
        clf = LinearSVC(dual=False)#, loss='hinge')
    #param_grid = {'C': [0.5, 1., 5., 10.]}
    #clf = GridSearchCV(svc, param_grid)
    # Check the training time for the SVC
    t=time.time()
    #clf.fit(X_train, y_train)
    scores = cross_val_score(clf, scaled_X, y, n_jobs=-1)
    clf.fit(scaled_X, y)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train classifier ...')
    print(scores)
    #print(clf.best_params_)
    #print(clf.best_score_)
    # Check the score of the SVC
    print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My classifier predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with classifier')

    dist_pickle = {}
    dist_pickle["classifier"] = clf
    dist_pickle["scaler"] = X_scaler
    dist_pickle["color_conv"] = COLOR_CONV
    dist_pickle["spatial_size"] = SPATIAL_SIZE
    dist_pickle["hist_bins"] = HIST_BINS
    dist_pickle["hist_range"] = HIST_RANGE
    dist_pickle["orient"] = ORIENTATION
    dist_pickle["pix_per_cell"] = PIX_PER_CELL
    dist_pickle["cell_per_block"] = CELL_PER_BLOCK
    dist_pickle["block_norm"] = BLOCK_NORM
    dist_pickle["hog_channel"] = HOG_CHANNEL
    dist_pickle["spatial_feat"] = SPATIAL_FEAT
    dist_pickle["hist_feat"] = HIST_FEAT
    dist_pickle["hog_feat"] = HOG_FEAT
    pickle.dump(dist_pickle, open(clf_pickle, "wb" ) ) 
    print('Classifier saved to pickle...') 
    
    return clf, X_scaler

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, clf, X_scaler, color_conv=None, 
              spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, 
              pix_per_cell=8, cell_per_block=2, block_norm="L2-Hys", hog_channel=0, 
              spatial_feat=True, hist_feat=True, hog_feat=True, draw_img=None, 
              cells_per_step=2):
    
    on_windows = []
    #draw_img = np.copy(img)
    #img = img.astype(np.float32)#/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_conv)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, block_norm,
                            vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, block_norm,
                            vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, block_norm,
                            vis=False, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                if not draw_img is None:
                    cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),6) 
                on_windows.append(bbox)
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def pipeline(image, return_all=False):
    box_list = []
    box_img = np.copy(image) if return_all else None
    for ystart, ystop, scale, cells_per_step in SEARCH_PARAMS:
        box_list.extend(find_cars(image, ystart, ystop, scale, clf, X_scaler, color_conv=COLOR_CONV, 
            spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE, orient=ORIENTATION, 
            pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, block_norm=BLOCK_NORM, 
            hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, 
            hog_feat=HOG_FEAT, draw_img=box_img, cells_per_step=cells_per_step))
    recent_boxes.append(box_list)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    for box_list in recent_boxes:
        heat = add_heat(heat,box_list)
    # Apply threshold to help remove false positives
    if len(recent_boxes) == recent_boxes.maxlen:
        threshold = THRESHOLD
    else:
        threshold = int(THRESHOLD / recent_boxes.maxlen * len(recent_boxes))
    heat = apply_threshold(heat, threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    env_img = np.copy(image)
    draw_labeled_bboxes(env_img, labels)
    #heatmap1 = np.zeros_like(image)
    #heatmap1[:,:,0] = (heatmap / heatmap.max() * 255).astype(np.int)
    #env_img = cv2.addWeighted(image, 1, heatmap1, 1, 0)
    if return_all:
        return box_list, box_img, heatmap, env_img
    return env_img

# MAIN

clf_pickle = CLASSIFIER + APPENDIX
print(clf_pickle)

if os.path.exists(clf_pickle):
    dist_pickle = pickle.load( open(clf_pickle, "rb" ) )
    clf = dist_pickle["classifier"]
    X_scaler = dist_pickle["scaler"]
    assert COLOR_CONV == dist_pickle["color_conv"]
    assert SPATIAL_SIZE == dist_pickle["spatial_size"]
    assert HIST_BINS == dist_pickle["hist_bins"]
    assert HIST_RANGE == dist_pickle["hist_range"]
    assert ORIENTATION == dist_pickle["orient"]
    assert PIX_PER_CELL == dist_pickle["pix_per_cell"]
    assert CELL_PER_BLOCK == dist_pickle["cell_per_block"]
    assert BLOCK_NORM == dist_pickle["block_norm"]
    assert HOG_CHANNEL == dist_pickle["hog_channel"]
    assert SPATIAL_FEAT == dist_pickle["spatial_feat"]
    assert HIST_FEAT == dist_pickle["hist_feat"]
    assert HOG_FEAT == dist_pickle["hog_feat"] 
    print('Classifier loaded from pickle...') 
else:
    clf, X_scaler = train()

# ## Process Video

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

video_input = 'project_video.mp4'
video_output = os.path.join('output_images', '{}_{}_{}_L{}_T{}{}'.format(
        video_input[:-4], CLASSIFIER, COLOR_CONV[4:] if COLOR_CONV else 'RGB', 
        LAST_N, THRESHOLD, video_input[-4:]))
#video_output = 'tmp.mp4'
#video_output = 'output_images/test_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip(video_input)#.subclip(27,28)
video_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)
