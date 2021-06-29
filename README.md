# Object Detection using Homography
This repo gives a demonstration as to how to detect any query objects which is part of the source image using the Feature matching method, and then finding the Homography. The code used, finds feature points in both the source and query images using feature descriptors like ORB, SIFT and the best pair of matches between them. 

In essence, the code returns the locations of a query object present in the source image which is sufficient to detect the query object exactly on the source Image.

Different query objects are extracted from the source image using Microsoft paint and the methodology is evaluated on all the query objects. Beacuse of matching features each between the extracted object and its source image, all the query objects were detected correctly. Objects which are part of the scene have much more matching lines drawn than objects which are not part of the scene.

## Design Pipeline
The Design Pipeline is as follows:
* Read the pair of source and query images.

### Source Image
![Source image](Source%20Image/scene2.png?raw=true)

### Query Object
![Query Object](Query%20Object/scene2_obj6.png?raw=true)

* Initiate a keypoint descriptor object. Here SIFT is used.
* Find the keypoints and descriptors with SIFT.
* Match object using FLANN based matcher.
* Retrieve the top two matches for each descriptor using KNN match with N=2.
* Store all the good matches as per Lowe's ratio test.
* Detect the keypoints for good matches only.
* Compute the homography between keypoints of matching pair sets using RANSAC.
* Perform a perspective transform on the query image.
* Detect the query object from the source image by drawing macthing lines between them.

### Detected Object
![Detected Object](/Detected%20Objects/scene2_obj6_homography.png?raw=true)
