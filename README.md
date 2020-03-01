# Optical- Flow

This Project contains implementation of KLT-Tracker from scratch. The main intent of this project is to write algorithm to track objects in motion.

![Output sample](https://github.com/arsjindal/Optical-flow/blob/master/Sample_ouput.gif)


### Prerequisites

Please install requirements for this project in virtual environmnet by running the following code.

```
pip install -r requirements.txt
```

### Installing/ Running

A step by step series of examples that tell you how to get a development env running

Please follow code below to run this project:

```
python3 object_tracker.py 'video path' 'folder name for saving extracted images'
```

For example

```
python3 object_tracker.py Videos/Easy.mp4 Easy
```


## Running the tests

* After running the above code, it will ask for no. of objects to track. Enter the number of objects and hit space.
* Create a bounding box and hit space. Do the same process for each object. 
* Computation will start and each frame will pop out for no. of features tracked in each frame. 
* It will create bounding box for each object in its new position.
* Resulting Video will be saved in Results folder.


## Coder

* **Neil Rodrigues** - *Project Partner* - [rodr651](https://github.com/rodri651)

