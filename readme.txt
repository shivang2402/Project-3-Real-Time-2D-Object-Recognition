Name: Shivang Patel

Operating System: macOS (Apple Silicon, Homebrew)
IDE: Terminal + clang++ (C++17)

Video Link: https://drive.google.com/file/d/1pUuyKR1Qcm0uSmvZBkKftqwSiDppgfr3/view?usp=sharing

Instructions for running:
  cd src && make objrec && cd ..
  ./bin/objrec data

  Pass a directory of images as argument. The system processes still images.

Keys:
  c = original color view
  t = thresholded view (Task 1, written from scratch)
  m = morphological cleanup view (Task 2, written from scratch)
  s = segmentation / colored region map (Task 3)
  f = features overlay with oriented bounding box and axis (Task 4)
  n = manual training mode, prompts for label (Task 5)
  b = batch auto-train all labeled images (Task 5)
  r = classify using nearest neighbor with scaled Euclidean distance (Task 6)
  k = classify using KNN (K=3) with scaled Euclidean distance (Extension)
  e = classify using ResNet18 CNN embeddings (Task 9)
  g = generate 2D PCA embedding plot (Extension, press e first)
  x = print confusion matrix (Task 7)
  p = save classification result images and print feature vectors
  a = save threshold, morph, segment, features views for all images
  w = save screenshot
  ] = next image
  [ = previous image
  q = quit

Extensions:
  1. Morphological filtering written from scratch (customErode, customDilate)
     This is a second task from scratch in addition to thresholding.
     Test: press m to see cleaned binary image.

  2. KNN classifier (K=3) as alternative distance metric
     Test: press k on any image to see KNN classification result.

  3. 2D embedding plot using PCA on ResNet18 embeddings
     Test: press e then g to generate and save the scatter plot.

  4. 9 object categories (4 more than required 5)
     Categories: triangle, squeegee, allenkey, chisel, keyfob, chair, mug, stand, desk

Time travel days used: 0
