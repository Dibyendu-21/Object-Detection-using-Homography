# Object Detection using Homography
This repo gives a demonstration as to how to detect any query objects which is part of the source image using the Feature matching method, and then finding the Homography. The code used, finds feature points in both the source and query images using feature descriptors like ORB, SIFT and the best pair of matches between them. 

In essence, the code returns the locations of a query object present in the source image which is sufficient to detect the query object exactly on the source Image.

Different query objects are extracted from the source image using Microsoft paint and the methodology is evaluated on all the query objects. Beacuse of matching features each between the extracted object and its source image, all the query objects were detected correctly.

