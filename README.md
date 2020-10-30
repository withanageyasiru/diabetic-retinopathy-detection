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
