# Face Recognition

This is a repositiry for showcase usage of Inception Resnet (V1), pretrained on VGGFace2. Implementation from Tim Esler's [github repo](https://github.com/timesler/facenet-pytorch)

And also includes an implementation of MTCNN for face detection, fastest from the available. 

## Getting Started

1. Install:
    ```bash
    # With pip:
    pip install facenet-pytorch
    
    # or clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    
    # or use a docker container (see https://github.com/timesler/docker-jupyter-dl-gpu):
    docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython
    ```
1. In python, import facenet-pytorch and instantiate models:
    ```python
    from facenet_pytorch import MTCNN, InceptionResnetV1
    
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
    
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    ```
1. Process an image:
    ```python
    from PIL import Image
    
    img = Image.open(<image path>)

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=<optional save path>)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    ```

See `help(MTCNN)` and `help(InceptionResnetV1)` for usage and implementation details.

## Example notebooks 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BeefMILF/facenet/blob/master/facenet.ipynb)

This notebook demonstrates the use of packages: 
1. facenet-pytorch
2. mtcnn 
3. sklearn 
4. albumentations

### *Complete detection and recognition pipeline*
In this notebook was introduced a complete example pipeline utilizing datasets, dataloaders, basic data augmentation, training classifier on top of resnets embeddings and face tracking in video streams. 
![](https://github.com/BeefMILF/facenet/blob/master/examples/images/1_aug.gif)

## Prerequisites

In order to run the example code in google colab you need to prepare separate folders for images dataset.  

```
Project facenet
+-- facenet.ipynb
+-- data
|   +-- test_images
    |   +-- person1
        |   +-- 1.png
        |   +-- 2.png
    |   +-- person2
        |   +-- 1.png
        |   +-- 2.png
|   +-- train_images
    |   +-- person1
        |   +-- 1.png
        |   +-- 2.png
    |   +-- person2
        |   +-- 1.png
        |   +-- 2.png
|   +-- test_images_cropped
    |   +-- person1
        |   +-- 1.png
        |   +-- 2.png
    |   +-- person2
        |   +-- 1.png
        |   +-- 2.png 
|   +-- train_images_cropped
    |   +-- person1
        |   +-- 1.png
        |   +-- 2.png
    |   +-- person2
        |   +-- 1.png
        |   +-- 2.png
```

Note, ```<images folder>_cropped``` folders are automatically generated in code. All images should be (.png, jpeg, jpg) and converted to RGB automatically.

Then, after preparing ```test_images``` and ```train_images```, we can easily apply face detection using MTCNN and save in ```<images folder>_cropped```. 

Following all the above, all cropped images can be ran through Inception Resnet model in order to get embeddings or probabilities. In our case, we are getting embeddings to train on them SVM classifier from sklearn (best parameters were found by SearchGrid and saved in ```data``` folder as ```svm.sav```). To make our classifier more stable, some augmentations were applied(you can observe them in notebook). 

All embeddings from images were saved in ```data``` folder as ```trainEmbeds.npz``` and ```testEmbeds.npz```.


## Face tracking in video streams 

Here we may see some obstacles, such as wrong-labelled classes and narrow-mindedness of our model(classifier predicts the most probable face among all known/trained, so it lacks of ability to distinguish known from unknown person, left 3 people were not in train dataset)

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

![](https://github.com/BeefMILF/facenet/blob/master/examples/images/1.jpeg)
![](https://github.com/BeefMILF/facenet/blob/master/examples/images/1_aug.gif)
![](https://github.com/BeefMILF/facenet/blob/master/examples/images/2.jpeg)
![](https://github.com/BeefMILF/facenet/blob/master/examples/images/2_aug.gif)
![](https://github.com/BeefMILF/facenet/blob/master/examples/videos/1.gif)
![](https://github.com/BeefMILF/facenet/blob/master/examples/videos/1_aug.gif)
![](https://github.com/BeefMILF/facenet/blob/master/examples/videos/2.gif)
![](https://github.com/BeefMILF/facenet/blob/master/examples/videos/2_aug.gif)
