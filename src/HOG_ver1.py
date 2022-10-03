import cv2
import numpy as np
import matplotlib.pyplot as plt
#need for math.sqrt and pow functions
import math

def get_differential_filter():
    filter_x = np.array([[0,0,0], [-1,0,1], [0,0,0]])
    filter_y = np.array([[0,-1,0], [0,0,0], [0,1,0]])

    return filter_x, filter_y


def filter_image(im, filter):

    shape = im.shape
    x_pixels = shape[0]
    y_pixels = shape[1]

    im_filtered = np.zeros((x_pixels, y_pixels))

    for y in range(y_pixels):
        for x in range(x_pixels):
            if(x == 0 or x == (x_pixels-1) or y == 0 or y == (y_pixels-1)):
                n=1
                #do something for edge cases (later)
            else:
                total = 0
                num_multiples = 0
                for i_y in range(3):
                    for i_x in range(3):
                        total += im[x+i_x-1, y+i_y-1] * filter[i_x,i_y]
                        num_multiples += abs(filter[i_x,i_y])

                #total = total/num_multiples
                im_filtered[x,y] = total

    return im_filtered


def get_gradient(im_dx, im_dy):

    if not (im_dx.shape == im_dy.shape):
        #I would throw an exception here and terminate but I dont want to import additional packages
        print("Error, mismatch image sizes")

    shape = im_dx.shape
    x_pixels = shape[0]
    y_pixels = shape[1]

    grad_mag = np.zeros((x_pixels, y_pixels))
    grad_angle = np.zeros((x_pixels, y_pixels))

    for y in range(y_pixels):
        for x in range(x_pixels):
            #if both x and y gradients are 0, angle and mag are 0
            if(im_dx[x,y] == 0) and (im_dy[x,y] == 0):
                grad_mag[x,y] = 0
                grad_angle[x,y] = 0
            #if x is 0 then the angle is 90
            elif(im_dx[x,y] == 0):
                grad_mag[x,y] = math.sqrt(math.pow(im_dx[x,y], 2) + math.pow(im_dy[x,y], 2))
                grad_angle[x,y] = 90
            #if y is 0 then the angle is 0
            elif(im_dy[x,y] == 0):
                grad_mag[x,y] = math.sqrt(math.pow(im_dx[x,y], 2) + math.pow(im_dy[x,y], 2))
                grad_angle[x,y] = 0
            else:
                grad_mag[x,y] = math.sqrt(math.pow(im_dx[x,y], 2) + math.pow(im_dy[x,y], 2))
                grad_angle[x,y] = math.degrees(math.atan(im_dy[x,y]/im_dx[x,y]))
                if(grad_angle[x,y] < 0):
                    grad_angle[x,y] += 180


    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):

    if not (grad_mag.shape == grad_angle.shape):
        #I would throw an exception here and terminate but I dont want to import additional packages
        print("Error, mismatch image sizes")

    shape = grad_mag.shape
    x_pixels = shape[0]
    y_pixels = shape[1]

    num_cells_x = int(x_pixels/cell_size)
    num_cells_y = int(y_pixels/cell_size)

    ori_histo = np.zeros((num_cells_x,num_cells_y,6))

    #for each cell build bins
    for y_cell_ct in range(num_cells_y):
        for x_cell_ct in range(num_cells_x):
            #for each pixel in each cell:
            for y in range(cell_size):
                for x in range(cell_size):

                    real_x = x + (x_cell_ct*cell_size)
                    real_y = y + (y_cell_ct*cell_size)

                    pixel_grad = grad_mag[real_x][real_y]
                    pixel_angle = grad_angle[real_x][real_y]

                    # if(pixel_angle < 30):
                    #     ori_histo[x_cell_ct][y_cell_ct][0] += pixel_grad
                    # elif(pixel_angle < 60):
                    #     ori_histo[x_cell_ct][y_cell_ct][1] += pixel_grad
                    # elif(pixel_angle < 90):
                    #     ori_histo[x_cell_ct][y_cell_ct][2] += pixel_grad
                    # elif(pixel_angle < 120):
                    #     ori_histo[x_cell_ct][y_cell_ct][3] += pixel_grad
                    # elif(pixel_angle < 150):
                    #     ori_histo[x_cell_ct][y_cell_ct][4] += pixel_grad
                    # else:
                    #     ori_histo[x_cell_ct][y_cell_ct][5] += pixel_grad

                    if(pixel_angle < 15):
                        ori_histo[x_cell_ct][y_cell_ct][0] += pixel_grad
                    elif(pixel_angle < 45):
                        ori_histo[x_cell_ct][y_cell_ct][1] += pixel_grad
                    elif(pixel_angle < 75):
                        ori_histo[x_cell_ct][y_cell_ct][2] += pixel_grad
                    elif(pixel_angle < 105):
                        ori_histo[x_cell_ct][y_cell_ct][3] += pixel_grad
                    elif(pixel_angle < 135):
                        ori_histo[x_cell_ct][y_cell_ct][4] += pixel_grad
                    elif(pixel_angle < 165):
                        ori_histo[x_cell_ct][y_cell_ct][5] += pixel_grad
                    else:
                        ori_histo[x_cell_ct][y_cell_ct][0] += pixel_grad

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do

    shape = ori_histo.shape
    num_cells_x = shape[0]
    num_cells_y = shape[1]

    ori_histo_normalized = np.zeros((int(num_cells_x-(block_size-1)), int(num_cells_y-(block_size-1)), int(6*(math.pow(block_size,2)))))

    #-1 to avoid going over edge since we will also be accessing the next cell over in ori_histo
    for y in range(num_cells_y-1):
        for x in range(num_cells_x-1):

            #ori_histo_normalized[x][y] = np.concatenate((ori_histo[x][y], ori_histo[x+1][y], ori_histo[x][y+1], ori_histo[x+1][y+1]), axis=0)

            for i in range(6):
                ori_histo_normalized[x][y][0+(4*i)] = ori_histo[x][y][i]
                ori_histo_normalized[x][y][1+(4*i)] = ori_histo[x+1][y][i]
                ori_histo_normalized[x][y][2+(4*i)] = ori_histo[x][y+1][i]
                ori_histo_normalized[x][y][3+(4*i)] = ori_histo[x+1][y+1][i]

            #normalizing
            #totaling
            total = 0
            for j in range(24):
                total += math.pow(ori_histo_normalized[x][y][j],2)
            total_norm = math.sqrt(total)

            for k in range(24):
                ori_histo_normalized[x][y][k] = ori_histo_normalized[x][y][k]/total_norm

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    filter_x, filter_y = get_differential_filter()

    im_filtered_x = filter_image(im, filter_x)
    im_filtered_y = filter_image(im, filter_y)

    grad_mag, grad_angle= get_gradient(im_filtered_x, im_filtered_y)

    ori_histo = build_histogram(grad_mag, grad_angle, 4)

    hog = get_block_descriptor(ori_histo, 2)

    # visualize to verify
    visualize_hog(im, hog, 4, 2)

    return hog

def extract_hog_no_vis(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    filter_x, filter_y = get_differential_filter()

    im_filtered_x = filter_image(im, filter_x)
    im_filtered_y = filter_image(im, filter_y)

    grad_mag, grad_angle= get_gradient(im_filtered_x, im_filtered_y)

    ori_histo = build_histogram(grad_mag, grad_angle, 8)

    hog = get_block_descriptor(ori_histo, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def merge_vectors_from_hog(hog):
    num_descriptors_x, num_descriptors_y, descriptor_len = hog.shape
    v = []
    for y in range(num_descriptors_y):
        for x in range(num_descriptors_x):
            for z in range(descriptor_len):
                v.append(hog[x,y,z])
    return(v)

def vector_norm(v):
    total = 0
    for i in range(len(v)):
        total += math.pow(v[i], 2)

    return(math.sqrt(total))

def vector_norm_each_element(v):
    total = 0
    for i in range(len(v)):
        total += v[i]

    mean = total/len(v)

    new_v = []

    for j in range(len(v)):
        new_v = v[j] - mean

    return(new_v)

def face_recognition(I_target, I_template):

    # convert grey-scale image to double format
    I_target = I_target.astype('float') / 255.0
    I_template = I_template.astype('float') / 255.0

    #get image sizes
    template_pixels_x, template_pixels_y = I_template.shape
    target_pixels_x, target_pixels_y = I_target.shape

    template_hog = extract_hog_no_vis(I_template)

    template_vector = merge_vectors_from_hog(template_hog)

    template_vector_norm = vector_norm(template_vector)

    template_descriptors_norm = vector_norm_each_element(template_vector)

    #array to hold bounding boxes
    bounding_boxes_list = []

    print(target_pixels_x-template_pixels_x,target_pixels_y-template_pixels_y)
    for y in range(target_pixels_y-template_pixels_y):
        for x in range(target_pixels_x-template_pixels_x):
            print(x,y)
            im_test = I_target[x:(x+template_pixels_x), y:(y+template_pixels_y)]

            test_hog = extract_hog_no_vis(im_test)
            test_vector = merge_vectors_from_hog(test_hog)
            test_vector_norm = vector_norm(test_vector)
            test_descriptor_norm = vector_norm_each_element(test_vector)

            s = (np.dot(template_descriptors_norm, test_descriptor_norm)) / (template_vector_norm * test_vector_norm)
            if(s >= 50):
                print(s)


    im_x_target = filter_image(I_target, filter_x)
    im_y_target = filter_image(I_target, filter_y)



    grad_mag_target, grad_angle_target= get_gradient(im_x_target, im_y_target)


    ori_histo_target = build_histogram(grad_mag_target, grad_angle_target, 8)


    hog_target = get_block_descriptor(ori_histo_target, 2)


    print(hog_template.shape)

    #for y_feat in range()

    bounding_boxes = np.array([[0,0,1]])
    return  bounding_boxes

def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('einstien.jpg', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
