# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import os, sys
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_action_sequence_panorama(images, num_fg_objs, display=False):
    ref = images[0]
    warped_list = [ref]
    print("Warping images")
    for im in tqdm(images[1:]):
        warped = warp(im, ref)
        warped_list.append(warped)
        
    bg = extract_background(warped_list, num_fg_objs)
    if display:
        imshow('background', bg)
        
    result = merge_foregrounds(bg, warped_list, num_fg_objs, display)
    return result
    
def get_edges(im, smooth_sigma=3.0, thresh_sigma=0.33):
    smoothed = cv2.GaussianBlur(im, (7,7), smooth_sigma)
    v = np.median(smoothed)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - thresh_sigma) * v))
    upper = int(min(359, (1.0 + thresh_sigma) * v))
    edged = cv2.Canny(smoothed, lower, upper)
    return edged

def get_contour(c, shape):
    ret = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(ret, c, -1, (255), 3)
    #struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    #ret = cv2.dilate(ret, kernel = struct_elt, iterations = 3)
    return ret

def hsv_dist(hsv0, hsv1):
    h0, s0, v0 = tuple(hsv0[...,i] for i in range(3))
    h1, s1, v1 = tuple(hsv1[...,i] for i in range(3))
    dh = np.min((np.abs(h1-h0), (np.ones(h0.shape) * 360.)-np.abs(h1-h0)), axis=0) / 180.0
    ds = np.abs(s1-s0)
    dv = np.abs(v1-v0) / 255.0
    distance = np.sqrt(dh*dh+ds*ds+dv*dv)
    return distance

def hue_dist(h0, h1):
    dist = np.min((np.abs(h1.astype(np.float64)-h0), 
                   (np.ones(h0.shape, dtype=np.float64) * 360.)
                   - np.abs(h1.astype(np.float64)-h0)), axis=0)
    return dist

def extract_background(images, num_fg_objs=1, display=False):
    print("Extracting background")   
    fimg = images[0]
    limg = images[-1]
    
    fimg_hsv = cv2.cvtColor(fimg, cv2.COLOR_BGR2HSV_FULL)
    limg_hsv = cv2.cvtColor(limg, cv2.COLOR_BGR2HSV_FULL)
    
    diff_hsv = cv2.absdiff(fimg_hsv, limg_hsv)
    diff_hsv[...,0] = hue_dist(fimg_hsv[...,0], limg_hsv[...,0])

    diff_gray = np.linalg.norm(diff_hsv, axis=2)

#    diff_gray = cv2.cvtColor(cv2.absdiff(fimg, limg), cv2.COLOR_BGR2GRAY)
    if display:
        imshow('diff', cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    
#    diff_flat = diff_gray.flatten()
#    diff_flat, df_sz = np.sort(diff_flat), diff_flat.size
#    diff_flat = diff_flat[np.nonzero(diff_flat)]
#    one_perc = df_sz // 100
#    diff_flat = diff_flat[one_perc:-one_perc]
#    df_sz = diff_flat.size
#    
#    diff_gaps = np.array([diff_flat[i + 1] - diff_flat[i] for i in xrange(df_sz) if i+1 < df_sz])
#    thresh = diff_flat[np.argmax(diff_gaps)]

    thresh = np.mean(diff_gray)
    
    _, diff_bin = cv2.threshold(diff_gray.astype(np.float32), thresh, 255, cv2.THRESH_BINARY)
    if display:
        imshow('diff binary mask', diff_bin)  

    struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    diff_eroded = cv2.erode(diff_bin, kernel=struct_elt, iterations = 3)
    diff_dilated = cv2.dilate(diff_eroded, kernel=struct_elt, iterations = 3)

    if display:
        imshow('dilated mask', diff_dilated)
    
    _, contours, _ = cv2.findContours(diff_dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # only keep N biggest 
    keep_num = 4*num_fg_objs + 1
    num_biggest = sorted(contours, key=cv2.contourArea, reverse=True)[:keep_num]
    
    if display:
        diff_biggest = np.zeros(diff_bin.shape, dtype=diff_bin.dtype)
        cv2.drawContours(diff_biggest, num_biggest, -1, (255), 3)
        imshow('biggest contours', diff_biggest) 
    
    diff_regions = [get_contour(c, diff_dilated.shape) for c in num_biggest]
    if display:
        for region in diff_regions:
            imshow('contour', region)     
    
    fedged = get_edges(fimg_hsv[...,0])
    ledged = get_edges(limg_hsv[...,0])
    
    if display:
        imshow('first image edges', fedged)
        imshow('last image edges', ledged) 
    
    edge_regions = [(np.bitwise_and(region, fedged), np.bitwise_and(region, ledged)) for region in diff_regions]
    region_nz = np.array([np.count_nonzero(region) for region in diff_regions])
    and_region_nz = np.array([[np.count_nonzero(a), np.count_nonzero(b)] for (a, b) in edge_regions])
    
    cocos = and_region_nz.astype(np.float) / np.tile(region_nz, (2, 1)).T
    if display:
        for i,er in enumerate(edge_regions):
            imshow('edge region A', er[0])
            imshow('edge region B', er[1])
            
    
    bg = images[0]
    for i, cc in enumerate(cocos):
        if np.allclose(cc[0], cc[1]):
            continue
        elif cc[0] > cc[1]:
            region = diff_regions[i]
            cv2.fillPoly(region, pts=[num_biggest[i]], color=(255))
            bg = blend(limg, bg, region)
            if display:
                imshow('filled region', region)
                
        
    return bg

def merge_foregrounds(bg, images, num_fg_objs=1, display=False, delay=100):
    print("Merging foregrounds")
    merged = bg.copy()
    fgs = []
    fimg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV_FULL)
    
    for it, limg in enumerate(tqdm(images)):
        limg_hsv = cv2.cvtColor(limg, cv2.COLOR_BGR2HSV_FULL)
        
        diff_hsv = cv2.absdiff(fimg_hsv, limg_hsv)
        diff_hsv[...,0] = hue_dist(fimg_hsv[...,0], limg_hsv[...,0])
    
        diff_gray = np.linalg.norm(diff_hsv, axis=2)
    
    #    diff_gray = cv2.cvtColor(cv2.absdiff(fimg, limg), cv2.COLOR_BGR2GRAY)
        if display:
            imshow('diff', cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            plt.pause(float(delay)/1000)
        
    #    diff_flat = diff_gray.flatten()
    #    diff_flat, df_sz = np.sort(diff_flat), diff_flat.size
    #    diff_flat = diff_flat[np.nonzero(diff_flat)]
    #    one_perc = df_sz // 100
    #    diff_flat = diff_flat[one_perc:-one_perc]
    #    df_sz = diff_flat.size
    #    
    #    diff_gaps = np.array([diff_flat[i + 1] - diff_flat[i] for i in xrange(df_sz) if i+1 < df_sz])
    #    thresh = diff_flat[np.argmax(diff_gaps)]
        
        thresh = np.mean(diff_gray)
        #print thresh
        
        _,diff_bin = cv2.threshold(diff_gray.astype(np.float32), thresh, 255, cv2.THRESH_BINARY)
    
        if display:
            imshow('diff binary mask', diff_bin)
            plt.pause(float(delay)/1000)
    
        struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        diff_eroded = cv2.erode(diff_bin, kernel=struct_elt, iterations = 1)
        diff_dilated = cv2.dilate(diff_eroded, kernel=struct_elt, iterations = 1)
    
        if display:
            imshow('dilated mask', diff_dilated)
            plt.pause(float(delay)/1000)
        
        _, contours, _ = cv2.findContours(diff_dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # only keep N biggest regions
        num_biggest = sorted(contours, key=cv2.contourArea, reverse=True)[:num_fg_objs]
        
        if display:
            diff_biggest = np.zeros(diff_bin.shape, dtype=diff_bin.dtype)
            cv2.drawContours(diff_biggest, num_biggest, -1, (255), 3)
            imshow('biggest contours', diff_biggest)
            plt.pause(float(delay)/1000)
        
        diff_regions = [get_contour(c, diff_dilated.shape) for c in num_biggest]
        if display:
            for region in diff_regions:
                imshow('contour', region)
                plt.pause(float(delay)/1000)
        
        fedged = get_edges(fimg_hsv[...,0])
        ledged = get_edges(limg_hsv[...,0])
        
        if display:
            imshow('first image edges', fedged)
            imshow('last image edges', ledged)
            plt.pause(float(delay)/1000)
              
        edge_regions = [(np.bitwise_and(region, fedged), np.bitwise_and(region, ledged)) for region in diff_regions]
        region_nz = np.array([np.count_nonzero(region) for region in diff_regions])
        and_region_nz = np.array([[np.count_nonzero(a), np.count_nonzero(b)] for (a, b) in edge_regions])
        
        cocos = and_region_nz.astype(np.float) / np.tile(region_nz, (2, 1)).T
        if display:
            for i,er in enumerate(edge_regions):
                imshow('edge region A', er[0])
                imshow('edge region B', er[1])
                plt.pause(float(delay)/1000)
        
        for i, cc in enumerate(cocos):
            if np.allclose(cc[0], cc[1]):
                continue
            elif cc[0] < cc[1]:
                region = diff_regions[i]
                cv2.fillPoly(region, pts=[num_biggest[i]], color=(255))
                struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
                region_dilated = cv2.dilate(region, kernel = struct_elt, iterations = 1)
                for fg in fgs:
                    region_dilated -= np.bitwise_and(region_dilated, fg)
                merged = blend(limg, merged, region_dilated)
                fgs.append(region)
                if display:
                    imshow('filled and dilated region', region_dilated)
                    plt.pause(float(delay)/1000)
        
    return merged

def blend(imA, imB, mask):
    mask = np.tile(mask, (3, 1, 1)).T.swapaxes(0,1)
    alpha = cv2.GaussianBlur(mask, (3, 3), 5)
    imA = imA.astype(np.float64)
    imB = imB.astype(np.float64)
     
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(np.float64)/255
    alpha_prime = np.ones(alpha.shape, dtype=np.float64) - alpha
    blended = (alpha * imA) + (alpha_prime * imB)
    return blended.astype(np.uint8)

def get_features(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    kps, des = surf.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, des
    
def match(imA, imB, ratio=0.7, reproj_thres=4):
    kpsA, featuresA = get_features(imA)
    kpsB, featuresB = get_features(imB)
    
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
    matches = []

    for m in rawMatches:
        # distance ratio test (Lowe's)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 3:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        
        if len(ptsA) > 3:
            H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thres)
            return H
        else:
            return None
    return None

def warp(imA, imB):
    H = match(imA, imB)
    dsize = (imA.shape[1], imA.shape[0])
    result = cv2.warpPerspective(imA, H, dsize)
    return result

def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))

def read_images(image_dir):
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(insensitive_glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]
    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))

    return images
    
def imshow(dispn, im):
    im2 = im.copy()
    if len(im2.shape) > 2:
        im2[:, :, 0] = im[:, :, 2]
        im2[:, :, 2] = im[:, :, 0]
    fig1, ax1 = plt.subplots()
    ax1.set_title(dispn)
    ax1.imshow(im2)
    plt.show()
    
def normalized_edges(im_in):
    image = im_in.copy()
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (7, 7), 0)     # blur image
    image = cv2.Canny(image, 75, 100)              # edge detection
    image = cv2.dilate(image, None, iterations=20) # dilation
    image = cv2.erode(image, None, iterations=20)  # erosion -- close gaps  
    return image
    
def normalize_shape(im_in):
    im_rsz = cv2.resize(im_in, (300, 300))
    grey = cv2.cvtColor(im_rsz, cv2.COLOR_BGR2GRAY) # gray    
    return grey

def detect_fg_objs(frames, min_area_px = 100, display=True):
    norms = list(map(normalize_shape, frames))
    average = np.sum(norms, axis=0, dtype=np.float32)/len(norms)
    averagecon = cv2.convertScaleAbs(average)
    grey = normalize_shape(frames[len(frames)//2])
    frameDelta = cv2.absdiff(averagecon, grey)
    #
    #frameDelta = cv2.bitwise_and(frameDelta, grey)
    thresh = cv2.threshold(frameDelta, 255//4, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)
    #
    if display:
        imshow('delta', frameDelta)
        imshow('thresh', thresh)
    image = normalized_edges(frameDelta)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    detects = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            bbox = cv2.boundingRect(c)
            detects.append(bbox)
    return detects

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def bb_center(bbox):
    x,y,w,h = bbox
    center = np.array((x+.5*w, y+.5*h))
    return center

def estimate_sample_rate(detects, frames, min_avg_dist_px = 50, display=True):
    frames = list(map(normalize_shape, frames))
    tracker = cv2.TrackerCSRT_create()
    start_pos = len(frames)//2
    areas = [w*h for (x,y,w,h) in detects]
    tracker.init(frames[start_pos], detects[np.argmax([areas])])
    tracks = []
    for f in frames[start_pos+1:]:
        (success, box) = tracker.update(f)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            tracks.append(box)
            if display:
                fc = f.copy()
                cv2.rectangle(fc, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imshow('tracking', fc)
    t0 = bb_center(tracks[0])
    i = 0
    cur_dist = -np.inf
    while cur_dist < min_avg_dist_px and i < len(tracks):
        tn = bb_center(tracks[i])
        cur_dist = dist(t0, tn)
        #print(cur_dist)
        i+=1
    return i
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates action sequence panoramas')
    parser.add_argument('data', help='Path to image folder or video file')
    parser.add_argument('-n', '--num-foreground', type=int, help='Number of foreground objects in scene (default: autodetect)')
    parser.add_argument('-r', '--rate', type=int, help='Frame resampling rate (e.g. 3 means keep 1 in 3 frames) (default: autodetect)')
    parser.add_argument('-s', '--scale', type=float, default=0.5, help='Scale input images by factor (default: %(default)s)')
    parser.add_argument('-o', '--output', default='out.png', help='Output filename (default: %(default)s)')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display intermediate results and verbose output')
    args = parser.parse_args()
    
    data_path = args.data
    scale = args.scale
    out_fname = args.output
    verbose = args.verbose
    
    if os.path.isdir(data_path):
        print("Loading images")
        images = read_images(data_path)
    elif os.path.isfile(data_path):
        images = []
        vidcap = cv2.VideoCapture(data_path)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            sys.stdout.write("\rLoading frames{}".format("."*int(np.log(count+1))))
            sys.stdout.flush()
            success, image = vidcap.read()
            if success:
                images.append(image)
            count += 1
        print()
    else:
        sys.exit('File not found')
    
    images = [cv2.resize(im, None, fx=scale, fy=scale) for im in images]
    detects = None
    if args.num_foreground:
        nfg = args.num_foreground
    else:
        print('Estimating number of foreground objects')
        detects = detect_fg_objs(images)
        nfg = len(detects)
    print("Foreground objects: {}".format(nfg))
    if args.rate:
        rate = args.rate
    else:
        if detects is None:
            detects = detect_fg_objs(images)
        print('Estimating sampling rate')
        rate = estimate_sample_rate(detects, images)
    print("Sampling rate: {}".format(rate))
    images = images[::rate]
    pan = make_action_sequence_panorama(images, nfg, display=verbose)
    cv2.imwrite(out_fname, pan)
    
    if args.show:
        imshow('Result', pan)