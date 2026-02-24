# Bruce Maxwell
# CS 5330 S24
# Tool for building a set of eigenimages from a collection of images

import sys
import os
import numpy as np
import cv2

# reshapes and normalizes a 1-D column image, works for eigenvectors,
# differential images, and original images
def viewColumnImage( column, orgrows, orgcols, name ):

    src = np.reshape( column, (orgrows, orgcols) )
    minval = np.min(src)
    maxval = np.max(src)

    src = 255 * ( (src - minval) / (maxval - minval) )
    view = src.astype( "uint8" )
    cv2.imshow( name, view )


# main function for building an eigenspace
def main(argv):

    # all of the image need to be the same size
    # use first image as the size for all the rest of the images
    orgrows = 0
    orgcolrs = 0

    # check for a directory path
    if len(argv) < 2:
        print("usage: python %s <directory name>" % (argv[0]) )
        return

    # grab the directory path
    srcdir = argv[1]

    # open the directory and get a file listing
    filelist = os.listdir( srcdir );

    # sort the filenames so we don't get a random ordering
    filelist.sort()

    buildmtx = True
    print("Processing srcdir")

    for filename in filelist:

        print("Processing file %s" % (filename) )
        suffix = filename.split(".")[-1]

        # make sure it's an image file
        if not ('pgm' in suffix or 'tif' in suffix or 'jpg' in suffix or 'png' in suffix ):
            continue

        # read the image
        src = cv2.imread( srcdir + "/" + filename )

        # make the image a single channel
        src = src[:,:,1] # keep the green channel

        # resize the long size of the image to 160
        resizeFactor =  160 / max( src.shape[0], src.shape[1] )

        # src.shape is rows [0] and cols [1]
        
        if orgrows == 0:
            # resize
            src = cv2.resize( src, ( int(src.shape[1]*resizeFactor), int(src.shape[0]*resizeFactor)), interpolation=cv2.INTER_AREA )
        else:
            src = cv2.resize( src, (orgcols, orgrows), interpolation=cv2.INTER_AREA ) # resize to match the first image
    
        # reshape the image to a single vector
        newImg = np.reshape( src, src.shape[0]*src.shape[1] )
        newImg = newImg.astype( "float32" ) # cast image to floats

        if buildmtx:
            # first image
            Amtx = newImg
            buildmtx = False
            orgrows = src.shape[0]
            orgcols = src.shape[1]
        else:
            Amtx = np.vstack( (Amtx, newImg) ) # vertically stacking the image

    # compute the mean of all images
    meanvec = np.mean( Amtx, axis=0 ) # computing the mean image over all rows (axis 0)

    # difference matrix
    Dmtx = Amtx - meanvec
    Dmtx = Dmtx.T # transpose the difference matrix so it is ( rows*cols x nImages ), each image is a column

    # call SVD
    U, s, V = np.linalg.svd( Dmtx, full_matrices = False )

    # compute eigenvalues
    eval = s**2 / (Dmtx.shape[0] - 1)

    print("top 10 eigenvalues: ", eval[0:10] )

    # look at the eigenvectors (which are images)
    for i in range(10):
        name = "eigenvector %d" % (i)
        viewColumnImage( U[:,i], orgrows, orgcols, name )
        cv2.moveWindow( name, 100 + i*170, 200 )

        
    # look at the  mean image
    viewColumnImage( meanvec, orgrows, orgcols, "Mean image" )
    cv2.moveWindow( "Mean image", 100, 0 )


    # project some images into a 6-dim eigenspace and the recreate the original images from 6 numbers
    nEVec = 6
    position = 0
    embeddings = []
    for index in [0, 1, 36, 37]:

        # get the differential image
        firstImage = np.array( Dmtx[:, index] )

        # project the image onto the first nEVec eigenvectors
        projection = np.dot( firstImage.T, U[:, 0:nEVec] )
        embeddings.append( projection )

        # print coefficients
        toprint = "Image %d: " % (index)
        for j in range(len(projection)):
            toprint += "%7.1f  " % (projection[j])
        print(toprint)
        

        # reproject from six coefficients back into an image
        recreated = projection[0] * U[:,0]
        for j in range(1, len(projection) ):
            recreated += projection[j] * U[:, j] # add coef x eigenvector

        # look at the recreated image
        # show the recreated original image (after adding back the mean image) note less noise
        name = "Recreated Original %d" % (index)
        viewColumnImage( recreated + meanvec, orgrows, orgcols, name )
        cv2.moveWindow(name, 100 + (position) * 200, 400 )

        # show the original image
        name = "Original %d" % (index)
        viewColumnImage( Amtx[index,:], orgrows, orgcols, name )
        cv2.moveWindow(name, 100 + (position) * 200, 600 )
        position += 1
    

    print("Distances:")
    print("    0          1          2          3")
    for i in range(len(embeddings)):
        s = "%d" % (i)
        for j in range(len(embeddings)):
            distance = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))
            s += " %10.2f" % (distance)
        print(s)
        
    cv2.waitKey(0)
    
    return


if __name__ == "__main__":
    main(sys.argv)
