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

### Example notebooks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BeefMILF/facenet/blob/master/facenet.ipynb)

### *Complete detection and recognition pipeline*
In this notebook was introduced a complete example pipeline utilizing datasets, dataloaders, basic data augmentation, training classifier on top of resnets embeddings and face tracking in video streams. 

![](https://github.com/BeefMILF/facenet/blob/master/examples/images/1_aug.gif)

![](https://github.com/BeefMILF/facenet/blob/master/examples/images/2_aug.gif)

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

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

