# Using ClipEncoder and Scikit-Learn for a 4 animal classification 

## Goal

Test to see if using a ClipEncoder with Scikit-Learn can produce similar results for a 4 class image classification problem as Keras/Tensorflow.

### TL;DR

The very best accuracy achieved from the Project leaderboard was `0.97560`.

Using the techniques outlined in this project, I was able to achieve  an accuracy of `0.9969`.

## Background

This dataset came from the OpenCV course in which students had to create a Keras model to classify 4 different animals.

* cow

* elephant

* horse

* spider

Using Keras to solve this problem took some to time to create and test.  

It also took a long time to execute mostly because I was running on a Mac with only  a cpu.

My first attempt produced a model with `0.88292` accuracy.  

This project will attempt to solve the same problem, but this time using the `CLIP Encoder` to encode the images and then use Scikit-Learn to perform the classification.

The very best predicted score was `0.97560`.


## Output

```text

Training the model

(3997, 512)
(3997,)
[0.99125    0.9875     0.99123905 0.99374218 0.99624531]
Accuracy: 0.99 (+/- 0.00)
Training time: 21.010899782180786
Validating the model
0.9961832061068703
Validation time: 4.0592122077941895
Testing the model
Testing time: 7.148894786834717
Accuracy: 99.69%

Confusion Matrix:
Predicted  cow  elephant  horse  spider
Actual                                 
cow        420         0      4       0
elephant     0       220      0       0
horse        1         0    498       0
spider       0         0      0     496
Total time: 32.23063111305237
```

## Submission Accuracy

For the submission test, I went through the Test images and manually created what I believe to be the correct answers. It is possible that the actual submission classifications could be different from mine, or I may have missed a few but I believe my true values to be accurate.

I did not want to submit my predictions to Kaggle because this was not done using the course restrictions of using Keras.

```text
Submission Accuracy: 99.69%

Confusion Matrix:
Predicted  cow  elephant  horse  spider
Actual                                 
cow        420         0      4       0
elephant     0       220      0       0
horse        1         0    498       0
spider       0         0      0     496

```

## Summary

From the experiment above, that using a ClipEncoder with Scikit-Learn for a classification problem as outlined above is a better approach than using deep learning techniques.

The approach is much simplier to understand, faster to setup, infinitely faster to execute and produces superior results.

## Resources

### YouTube

https://youtu.be/lzXKsY3bANw?si=OX2WVuTblQTJ84sO

### Github

https://github.com/probabl-ai/youtube-appendix/blob/main/01-sklearn-image/notebook.ipynb

### PyPI

https://pypi.org/project/embetter/

which has some invaluable embeddings specifically designed for the scikit-learn ecosystem.

After walking through the resources, I decided to try it on the `Cats vs Dogs` ( or is it `Dogs vs Cats`) dataset which I downloaded years ago.  I had worked through this dataset using Tensorflow/Keras back in the day.

To my surprise, scikit-learn with the embetter image embeddings did surprisingly well.

Using a naive LogisticRegression classifier, the model had an average cross validation accuracy score of `0.9956338874424192`.

Testing this on a holdout dataset of 20 cat images and 20 dog images, it was able to classify all of the holdouts correctly.

## What is a ClipEncoder

### CLIP ( Contrastive Language–Image Pretraining ) Background Information

**What is a CLIP encoder?**

CLIP stands for Contrastive Language–Image Pretraining. It’s a model made by OpenAI that can understand both images and text, and match them to each other. So, if you show CLIP a picture of a dog and the word “dog,” it will know they go together.

CLIP uses two main parts:

An image encoder – that looks at an image and turns it into numbers (called an embedding).
A text encoder – that does the same thing for text.

**What does it mean to "encode" an image?**

When we encode an image, we’re turning it into a list of numbers that represents the important stuff about the image—kind of like its fingerprint.

This list of numbers is called a vector or embedding. It's like a summary of the image that CLIP can use to compare it with other images or text.

Think of it like this:

Imagine you take a picture of a cat.
The CLIP image encoder looks at that picture and gives you a list of, say, 512 numbers.
These numbers don't look like much to us, but to the model, they capture key features like shapes, colors, and what objects are in the image.

**What does the output represent?**

The output is a list of numbers (a vector)—for example:

[0.12, -0.58, 0.33, ..., 0.05]  ← 512 numbers
Each number in that list represents a different feature or pattern in the image. Alone, they don’t mean much to humans, but together, they help the computer know what’s in the image.

**For example:**

Similar images (like two pictures of dogs) will have similar vectors.
Different images (like a dog vs. a car) will have different vectors.

**Why is this useful?**

Because once an image is a vector:

You can compare it with text vectors (like the word "cat" or "dog").
You can search for images that are similar.
You can do things like captioning, clustering, or even generating images based on text.

**Summary**

A CLIP encoder turns images into numbers (vectors).
These numbers summarize what’s in the image.
They help computers understand and compare images and text—even if the computer has never seen the exact image before.


**Here’s how it works:**

When you pass an image to CLIP:

The image is loaded (usually as pixels).
It’s resized and preprocessed (to match what CLIP expects—like 224×224 pixels).
Then it's passed through the image encoder (like a modified ResNet or Vision Transformer).
The encoder outputs a vector of numbers (the embedding) that represents only the visual content of the image.




