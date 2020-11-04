# Diabetic retinopathy detection

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Diabetic retinopathy is a disease caused due to diabetes.
  - Leakage and swelling of blood vessels of retina.
  - Sometimes abnormal blood vessel growth.
  - Common symptoms include blurred vision, spots in vision and complete vision loss.
  - Early detection is done through analyzing eye fundus image with the aid of customized software.
  - It is classified as mild, moderate, severe, very severe and proliferative.
  - Early detection can reduce the chances of vision loss by 90%


# New Features!

  - Incresed accuracy


You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop markdown and HTML files into Dillinger
  - Export documents as Markdown, HTML and PDF


> Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people who have diabetes. It affects blood vessels in the retina (the light-sensitive layer of tissue in the back of your eye).  


### Tech


Dillinger uses a number of open source projects to work properly:

* [TensorFlow] - HTML enhanced for web apps!
* [Open CV] - awesome web-based text editor


And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

### Installation

## Environment Setup Guide (Conda)


Then make a conda environment using the Anaconda prompt

```bash
conda create -n name_for_the_environment python=3.8
```

After creating, activate it,
```bash
name_of_environment python=3.8
```

Then go to the location where your project wants to create and make a file named as requirements.txt and copy these.
  * matplotlib 
  * opencv-contrib-python==3.4.8.29
  * opencv-python==3.4.8.29
  * tensorflow==2.2.0

Then run the followning command using package manager [pip](https://pip.pypa.io/en/stable/)

```bash
pip install -r requirements.txt
```

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ NODE_ENV=production node app
```

### Plugins

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| ResNet | [https://keras.io/api/applications/resnet] |


### Development

Want to contribute? Great!


License
----

MIT



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [Open CV]: <https://opencv-python-tutroals.readthedocs.io/en/latest/#>
   [TensorFlow]: <https://www.tensorflow.org/tutorials>
   

