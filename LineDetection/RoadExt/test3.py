import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip



def project_lane_line(original_image, binary_warped, ploty, left_fitx, right_fitx, m_inv, fill = True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    """
    line_y = original_image.shape[0] // 2  # Middle y-coordinate
    line_x = (left_fitx[line_y] + right_fitx[line_y]) // 2  # Middle x-coordinate
    pt1 = (int(line_x), int(line_y))
    pt2 = (int(line_x), original_image.shape[0])    
    cv2.line(original_image, pt1, pt2, (255, 0, 0), thickness=3)    
    """

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image

    if(fill):
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    else:
        cv2.polylines(color_warp, np.int32([pts]), isClosed=False, color=(0, 255, 0), thickness=3)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (original_image.shape[1], original_image.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result


def seperate_hls(rgb_img):
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    return h, l, s

def seperate_lab(rgb_img):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    return l, a, b

def seperate_luv(rgb_img):
    luv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Luv)
    l = luv[:,:,0]
    u = luv[:,:,1]
    v = luv[:,:,2]
    return l, u, v

def binary_threshold_lab_luv(rgb_img, athresh, bthresh, lthresh):
    l, a, b = seperate_lab(rgb_img)
    l2, u, v = seperate_luv(rgb_img)

    # cv2.imshow("l_channel",l)
    # cv2.imshow("a_channel",a)
    # cv2.imshow("b_channel",b)
    # cv2.imshow("l2_channel",l2)
    # cv2.imshow("u_channel",u)
    # cv2.imshow("v_channel",v)

    binary = np.zeros_like(l)
    binary[
        # ((b > bthresh[0]) & (b <= bthresh[1])) |
        ((b > athresh[0]) & (b <= athresh[1])) &
        ((b > bthresh[0]) & (b <= bthresh[1])) &
        ((l2 > lthresh[0]) & (l2 <= lthresh[1]))
    ] = 1
    

    # binary_a = np.zeros_like(l)
    # binary_b = np.zeros_like(l)
    # binary_l2 = np.zeros_like(l)
    # binary_a[
    #     ((a > athresh[0]) & (a <= athresh[1]))
    # ] = 1
    # binary_b[
    #     ((b > bthresh[0]) & (b <= bthresh[1]))
    # ] = 1
    # binary_l2[
    #     ((l2 > lthresh[0]) & (l2 <= lthresh[1]))
    # ] = 1
    # cv2.imshow("a_thold",binary_a*255)
    # cv2.imshow("b_thold",binary_b*255)
    # cv2.imshow("l2_thold",binary_l2*255)
    
    return binary

def binary_threshold_hls(rgb_img, sthresh, lthresh):
    h, l, s = seperate_hls(rgb_img)
    binary = np.zeros_like(h)
    binary[
        ((s > sthresh[0]) & (s <= sthresh[1])) &
        ((l > lthresh[0]) & (l <= lthresh[1]))
    ] = 1
    return binary

def gradient_threshold(channel, thresh):
    # Take the derivative in x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold gradient channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def cut_out_line(original_image, binary_warped, ploty, left_fitx, right_fitx, m_inv):
    # Create a blank image with the same size as binary_warped
    cutout = np.zeros_like(binary_warped).astype(np.uint8)

    """
    line_y = original_image.shape[0] // 2  # Middle y-coordinate
    line_x = (left_fitx[line_y] + right_fitx[line_y]) // 2  # Middle x-coordinate
    pt1 = (int(line_x), int(line_y))
    pt2 = (int(line_x), original_image.shape[0])    
    cv2.line(original_image, pt1, pt2, (255, 0, 0), thickness=3)    
    """
        
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the blank image
    cv2.fillPoly(cutout, np.int_([pts]), 255)
    
    # Warp the cutout back to original image space using inverse perspective matrix (Minv)
    newcutout = cv2.warpPerspective(cutout, m_inv, (original_image.shape[1], original_image.shape[0]))
    
    # Apply bitwise AND operation between the original image and the cutout to get the lane region
    result = cv2.bitwise_and(original_image, original_image, mask=newcutout)
    return result

IMG_SIZE = np.array((224, 224, 3))[::-1][1:]#frame.shape[::-1][1:]
OFFSET = 20


PRES_SRC_PNTS = np.float32([
    (10, 100), # Top-left corner
    (0, 223), # Bottom-left corner
    (223, 223), # Bottom-right corner
    (224-50, 100) # Top-right corner
])



PRES_DST_PNTS = np.float32([
    [OFFSET, 0], 
    [OFFSET, IMG_SIZE[1]],
    [IMG_SIZE[0]-OFFSET, IMG_SIZE[1]], 
    [IMG_SIZE[0]-OFFSET, 0]
])


N_WINDOWS = 10
MARGIN = 100
RECENTER_MINPIX = 50

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720 # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700 # meters per pixel in x dimension

def histo_peak(histo, threshold = 10):
    """Find left and right peaks of histogram"""
    midpoint = np.int(histo.shape[0]/2)
    leftx_base = np.argmax(histo[:midpoint]> threshold)
    rightx_base = np.argmax(histo[midpoint:]> threshold) + midpoint
    debug_var = histo.shape
    debug_var = np.argmax(histo)
    # debug_var = np.argmax(histo)
    return leftx_base, rightx_base
def check_same_sign(lst):
    return all(x >= 0 for x in lst) or all(x < 0 for x in lst)
def get_lane_indices_sliding_windows(binary_warped, leftx_base, rightx_base, n_windows, margin, recenter_minpix,ax):
    """Get lane line pixel indices by using sliding window technique"""
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.copy()
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/n_windows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    switch_left = False
    #print(leftx_current, rightx_current)
    moved_right = []
    margin_right = margin
    margin_left = margin +10
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin_left
        win_xleft_high = leftx_current + margin_left
        win_xright_low = rightx_current - margin_right
        win_xright_high = rightx_current + margin_right

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)

        #ax.clear()
        #ax.imshow(out_img)
        #plt.draw()
        #plt.pause(0.001)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position

        #if len(good_left_inds[:good_left_inds//2])
        non_zero_right = nonzerox[good_right_inds]
        non_zero_left = nonzerox[good_left_inds]
        # Calculate the sum of the first half of nonzerox
        non_zero_right_sum = np.sum(non_zero_right)
        non_zero_left_sum = np.sum(non_zero_left)


       
        if(non_zero_right_sum==0):
            #print("The ones are evenly distributed or there are no ones.")
            rightx_current = rightx_current+ 30 if  rightx_current+ 30 <224 else rightx_current- 30 
            #window-=1
        
            #window-=1
        if(non_zero_left_sum == 0):
            leftx_current +=2
        


        if len(good_left_inds) > recenter_minpix-20:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        
        

            
        if len(good_right_inds) > recenter_minpix:        
            new_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            rightx_current = new_rightx_current 
            margin_right -=1


            

           
           
        
    #print()
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #print(left_lane_inds)
    return left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img

def get_lane_indices_from_prev_window(binary_warped_img, left_fit, right_fit, margin):
    """Detect lane line by searching around detection of previous sliding window detection"""
    nonzero = binary_warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if(len(lefty) ==0 or len(leftx) == 0):
        lefty = righty
        leftx = rightx[::-1] - 150
       

    if(len(rightx) ==0 and len(righty) == 0):
        
        rightx = leftx + 150
        righty = lefty

    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_lane_inds, right_lane_inds, ploty, left_fitx, right_fitx

GRADIENT_THRESH = (10, 80)
# S_CHANNEL_THRESH = (80, 255)
# L_CHANNEL_THRESH = (80, 255)
A_CHANNEL_THRESH = (127 - 10, 127 + 10)
B_CHANNEL_THRESH = (127 - 10, 127 + 10)
L2_CHANNEL_THRESH = (150, 255)
video = cv2.VideoCapture('LineDetection/output.ogg')


import os
path = "./LineDetection/data/balanced_LOS/"
images = os.listdir(path)
images

for image in images:
    # Read the current frame
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        exit()

    frame = cv2.imread(path + image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    s_binary = binary_threshold_lab_luv(frame, A_CHANNEL_THRESH, B_CHANNEL_THRESH, L2_CHANNEL_THRESH)

        # Gradient threshold on S channel
    h, l, s = seperate_hls(frame)
    sxbinary = gradient_threshold(s, GRADIENT_THRESH)

        # Combine two binary images to view their contribution in green and red
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary))) * 255

    cv2.imshow("s_binary",s_binary*255)
    cv2.imshow("sxbinary",sxbinary*255)
    cv2.imshow("color_binary",color_binary)
    cv2.waitKey(0)
    M = cv2.getPerspectiveTransform(PRES_SRC_PNTS, PRES_DST_PNTS)
    M_INV = cv2.getPerspectiveTransform(PRES_DST_PNTS, PRES_SRC_PNTS)
    warped = cv2.warpPerspective(frame, M, IMG_SIZE, flags=cv2.INTER_LINEAR)
    warped_cp = warped.copy()
    #warped_poly = cv2.polylines(warped_cp, np.int32([PRES_DST_PNTS]), True, (255,0,0), 3)
    
    
    # Warp binary image of lane line
    binary_warped = cv2.warpPerspective(s_binary, M, IMG_SIZE, flags=cv2.INTER_LINEAR)
    #binary_warped2 = cv2.warpPerspective(s_binary2, M, IMG_SIZE, flags=cv2.INTER_LINEAR)
    
    # Calculate histogram of lane line pixels
    #print(binary_warped.shape)
    histogram = np.sum(binary_warped[200:,:], axis=0)
    #histogram2 = np.sum(binary_warped2[int(binary_warped2.shape[0]/2):,:], axis=0)
    leftx_base, rightx_base = histo_peak(histogram)
    left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img = get_lane_indices_sliding_windows(
        binary_warped, leftx_base, rightx_base, 16, 30, 40, None)
    if(len(left_lane_inds) <60):
        left_lane_inds = []
       
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #print(lefty, leftx)


    if(len(lefty) ==0 and len(leftx) == 0 and len(rightx) ==0 and len(righty) == 0):
        righty = np.array([223, 193])
        #rightx = np.array([112, 223])
    if(len(lefty) ==0 or len(leftx) == 0):
        lefty = righty
        leftx = rightx[::-1] - 150

    if(len(rightx) ==0 or len(righty) == 0):
        rightx = leftx + 150
        righty = lefty
    
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img_cp = out_img.copy()
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    #Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-MARGIN, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+MARGIN, 
                                ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-MARGIN, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+MARGIN, 
                                ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    result = project_lane_line(frame, binary_warped, ploty, left_fitx, right_fitx, M_INV)
    result2 = cut_out_line(frame, binary_warped, ploty, left_fitx, right_fitx, M_INV)

    cv2.imshow("result",result)
    #ax.clear()
    #ax.imshow(out_img_cp)
    cv2.imshow("binary_warped",binary_warped*255)

    
    #ax.plot(histogram)

    

    # Redraw the plot
    #plt.draw()
    #plt.pause(0.001)
    if(cv2.waitKey(0) == 'q'):
        break

cv2.destroyAllWindows()