# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import os, sys
from glob import glob

def make_action_sequence_panorama(images, num_fg_objs):
    ref = images[0]
    warped_list = [ref]
    for im in images[1:]:
        warped = warp(im, ref)
        warped_list.append(warped)
        
    bg = extract_background(warped_list, num_fg_objs)
    cv2.imshow('background', bg)
    cv2.waitKey(0)
    result = merge_foregrounds(bg, warped_list, num_fg_objs, display=False)
    return result
    
def get_edges(im, smooth_sigma=3.0, thresh_sigma=0.33):
    smoothed = cv2.GaussianBlur(im, (7,7), smooth_sigma);
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
    
    fimg = images[0]
    limg = images[-1]
    
    fimg_hsv = cv2.cvtColor(fimg, cv2.COLOR_BGR2HSV_FULL)
    limg_hsv = cv2.cvtColor(limg, cv2.COLOR_BGR2HSV_FULL)
    
    diff_hsv = cv2.absdiff(fimg_hsv, limg_hsv)
    diff_hsv[...,0] = hue_dist(fimg_hsv[...,0], limg_hsv[...,0])

    diff_gray = np.linalg.norm(diff_hsv, axis=2)

#    diff_gray = cv2.cvtColor(cv2.absdiff(fimg, limg), cv2.COLOR_BGR2GRAY)
    if display:
        cv2.imshow('diff', cv2.normalize(diff_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.waitKey(0)
    
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
    #cv2.imwrite('diff_bin.png', cv2.normalize(diff_bin, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    if display:
        cv2.imshow('diff binary mask', diff_bin)
        cv2.waitKey(0)

    struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    diff_eroded = cv2.erode(diff_bin, kernel=struct_elt, iterations = 3)
    diff_dilated = cv2.dilate(diff_eroded, kernel=struct_elt, iterations = 3)

    if display:
        cv2.imshow('dilated mask', diff_dilated)
        cv2.waitKey(0)
    
    contours, heirarchy = cv2.findContours(diff_dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # only keep N biggest 
    keep_N = 4*num_fg_objs + 1
    nbiggest = sorted(contours, key=cv2.contourArea, reverse=True)[:keep_N]
    
    if display:
        diff_biggest = np.zeros(diff_bin.shape, dtype=diff_bin.dtype)
        cv2.drawContours(diff_biggest, nbiggest, -1, (255), 3)
        cv2.imshow('biggest contours', diff_biggest)
        cv2.waitKey(0)
    
    diff_regions = [get_contour(c, diff_dilated.shape) for c in nbiggest]
    if display:
        for region in diff_regions:
            cv2.imshow('contour', region)
            cv2.waitKey(0)
    
    fedged = get_edges(fimg_hsv[...,0])
    ledged = get_edges(limg_hsv[...,0])
    
    if display:
        cv2.imshow('first image edges', fedged)
        cv2.imshow('last image edges', ledged)
        cv2.waitKey(0)
    
    
    edge_regions = [(np.bitwise_and(region, fedged), np.bitwise_and(region, ledged)) for region in diff_regions]
    
    region_nz = np.array([np.count_nonzero(region) for region in diff_regions])
    #print region_nz
    and_region_nz = np.array([[np.count_nonzero(a), np.count_nonzero(b)] for (a, b) in edge_regions])
    #print and_region_nz
    
    cocos = and_region_nz.astype(np.float) / np.tile(region_nz, (2, 1)).T
    if display:
        for i,er in enumerate(edge_regions):
            cv2.imshow('edge region A', er[0])
            cv2.imshow('edge region B', er[1])
            cv2.waitKey(0)
    
    bg = images[0]
    for i, cc in enumerate(cocos):
        if np.allclose(cc[0], cc[1]):
            continue
        elif cc[0] > cc[1]:
            region = diff_regions[i]
            cv2.fillPoly(region, pts=[nbiggest[i]], color=(255))
            bg = blend(limg, bg, region)
            if display:
                cv2.imshow('filled region', region)
                cv2.waitKey(0)
        
    return bg

def merge_foregrounds(bg, images, num_fg_objs=1, display=False, delay=100):
    print("Merging foregrounds")
    merged = bg.copy()
    fgs = []
    fimg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV_FULL)
    
    for it, limg in enumerate(images):
        print("# {}".format(it))
        limg_hsv = cv2.cvtColor(limg, cv2.COLOR_BGR2HSV_FULL)
        
        diff_hsv = cv2.absdiff(fimg_hsv, limg_hsv)
        diff_hsv[...,0] = hue_dist(fimg_hsv[...,0], limg_hsv[...,0])
    
        diff_gray = np.linalg.norm(diff_hsv, axis=2)
    
    #    diff_gray = cv2.cvtColor(cv2.absdiff(fimg, limg), cv2.COLOR_BGR2GRAY)
        if display:
            cv2.imshow('diff', cv2.normalize(diff_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            cv2.waitKey(delay)
        
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
            cv2.imshow('diff binary mask', diff_bin)
            cv2.waitKey(delay)
    
        struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        diff_eroded = cv2.erode(diff_bin, kernel=struct_elt, iterations = 1)
        diff_dilated = cv2.dilate(diff_eroded, kernel=struct_elt, iterations = 1)
    
        if display:
            cv2.imshow('dilated mask', diff_dilated)
            cv2.waitKey(delay)
        
        contours, heirarchy = cv2.findContours(diff_dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # only keep N biggest regions
        nbiggest = sorted(contours, key=cv2.contourArea, reverse=True)[:num_fg_objs]
        
        if display:
            diff_biggest = np.zeros(diff_bin.shape, dtype=diff_bin.dtype)
            cv2.drawContours(diff_biggest, nbiggest, -1, (255), 3)
            cv2.imshow('biggest contours', diff_biggest)
            cv2.waitKey(delay)
        
        diff_regions = [get_contour(c, diff_dilated.shape) for c in nbiggest]
        if display:
            for region in diff_regions:
                cv2.imshow('contour', region)
                cv2.waitKey(delay)
        
        fedged = get_edges(fimg_hsv[...,0])
        ledged = get_edges(limg_hsv[...,0])
        
        if display:
            cv2.imshow('first image edges', fedged)
            cv2.imshow('last image edges', ledged)
            cv2.waitKey(delay)
        
        
        edge_regions = [(np.bitwise_and(region, fedged), np.bitwise_and(region, ledged)) for region in diff_regions]
        
        region_nz = np.array([np.count_nonzero(region) for region in diff_regions])
        #print region_nz
        and_region_nz = np.array([[np.count_nonzero(a), np.count_nonzero(b)] for (a, b) in edge_regions])
        #print and_region_nz
        
        cocos = and_region_nz.astype(np.float) / np.tile(region_nz, (2, 1)).T
        if display:
            for i,er in enumerate(edge_regions):
                cv2.imshow('edge region A', er[0])
                cv2.imshow('edge region B', er[1])
                cv2.waitKey(delay)
        
        for i, cc in enumerate(cocos):
            if np.allclose(cc[0], cc[1]):
                continue
            elif cc[0] < cc[1]:
                region = diff_regions[i]
                cv2.fillPoly(region, pts=[nbiggest[i]], color=(255))
                struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
                region_dilated = cv2.dilate(region, kernel = struct_elt, iterations = 1)
                for fg in fgs:
                    region_dilated -= np.bitwise_and(region_dilated, fg)
                merged = blend(limg, merged, region_dilated)
                fgs.append(region)
                if display:
                    cv2.imshow('filled and dilated region', region_dilated)
                    cv2.waitKey(delay)
        
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
    surf = cv2.SURF()
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

#        pts_dist = np.array([np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) for (a, b) in zip(ptsA, ptsB)])
#        min_dist = np.median(pts_dist)
#        keep_idcs = np.where(pts_dist <= min_dist * 2)
#        
#        ptsA = ptsA[keep_idcs]
#        ptsB = ptsB[keep_idcs]
        
        if len(ptsA) > 3:
            H, s = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thres)
            return H
        else:
            return None
    return None

def warp(imA, imB):
    H = match(imA, imB)
    dsize = (imA.shape[1], imA.shape[0])
    result = cv2.warpPerspective(imA, H, dsize)
    return result
    
def read_images(image_dir):
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]
    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))

    return images
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate action sequence panorama')
    parser.add_argument('data', help='Path to image folder or video file')
    parser.add_argument('-fgo', '--foreground-objects', type=int, default=1, help='Number of foreground objects in scene (default: %(default)s)')
    parser.add_argument('-s', '--scale', type=float, default=0.2, help='Scale input images by factor (default: %(default)s)')
    parser.add_argument('-r', '--rate', type=int, default=3, help='Frame resampling rate (e.g. 3 means keep 1 in 3 frames) (default: %(default)s)')
    args = parser.parse_args()
    
    data_path = args.data
    scale = args.scale
    rate = args.rate
    
    if os.path.isdir(data_path):
        images = read_images(data_path)
        images = images[::rate]
    elif os.path.isfile(data_path):
        images = []
        vidcap = cv2.VideoCapture(data_path)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            if count % rate == 0:
                images.append(image)
            count += 1
    else:
        sys.exit('File not found')
    
    images = [cv2.resize(im, None, fx=scale, fy=scale) for im in images]
    
    pan = make_action_sequence_panorama(images, args.foreground_objects)
    cv2.imwrite('pano.png', pan)
    
    cv2.imshow('Sequence', pan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
