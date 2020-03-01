#######################################################

Project3B: OPTICAL FLOW

By: Aayush Jindal & Neil Rodrigues

######################################################

Please run the following commands in terminal for execution:

	python3 object_tracker.py

* Resulting Video are stored in "Results" Folder.

* Enter no. of objects to be tracked and select objects in popped up frame. 

* Then features are extracted using harris corner detection. These features are then processed to output new position of each feature in next frame using "estimatefeatureTranslation.py". These new position for each feature are extracted and stored in "estimateAllTranslation.py".
Then through these features, new bounding box coordinates are transformed using "applyGeometricTransformations.py". 

Please contact aayushj@seas.upenn.edu / rodri651@seas.upenn.edu in case of any confusion.

 

