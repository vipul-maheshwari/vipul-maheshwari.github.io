---
layout: post
title: Neural Style Transfer
date: '2022-03-03 22:12:00 +0530'
categories: others
published: true
permalink: /:categories/:title
---

In the next couple of minutes, you may master the use of one of the most important applications of Computer Vision. I will start from the very simple foundation of what Neural style transfer is and then slowly I’ll undergo what it precisely means and why it really works. I’ll explain the complete theory behind it.

*Note*: This is NOT a research based tutorial on implementing Neural Style Transfer. There are already a plethora of tutorials, blog posts and code implementations available on the Internet for that. Performing a single Google search will lead you to them. This article is all about understanding the practical use of Neural Style Transfer and it's applications. It might take you a couple of minutes to read this article completely *if you remain focused*, but it will cover everything that your boring professor in the class will talk about and that might take you couple of classes to understand it completely.

--------------------------------------------------------------------------------------------------------------------

***🧐 What is Neural Style Transfer and why it so important?***

&nbsp;
Deep learning being the modest and most important technologies in the world have already been used to create a list of variety of applications like object recognition, image processing, video processing, and many more. However, the most important application of deep learning is the one that is used to create artworks. And when an [AI generated image](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx) is sold for whooping $432,500 at an auction, it was pretty clear that AI will be the part of artworks in future.

*according to the wiki: Neural Style Transfer* refers to manipulating digital images, or videos, to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks in order to perform the image transformation.

In layman terms, The technique of merging style from one image into another while keeping the other's content intact is known as Neural style transfer. The only difference is the content image's style settings.

Common uses for NST are the creation of artificial artworks from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs. Several notable mobile apps use NST techniques for this purpose, including DeepArt and Prisma. This method has been used by artists and designers around the globe to develop new artwork based on existent styles. The NST algorithm is also used to create new images for video editing, for example by transferring the appearance of a video to a new video.

![nstexample](/assets_for_posts/deeplearning/What_is_neural_style_transfer/nstexample.jpeg)

With the exemplary growth in the hardware and AI enabled workflow architectures, style transfer can now be applied to captured and live video transcripts. Exponential jump in the number of videos that can be processed by a single machine is a key factor in the success of style transfer as it opens a whole new dimension for designers, content creators and the developers.

Apps like Prisma and DeepArt use NST to create new artworks from existing images. The list goes on for the different segments of Gaming, Commercial artworks, Virtual reality, metaverse and WEB 3.o universe.

--------------------------------------------------------------------------------------------------------------------

***🤔 How does Neural Style Transfer works?***

&nbsp;
Ok till now, we have a brief understanding of what NST is, where it is primarily used and how it works. Now, we will go through the details of how it works. In this particular section, we will examine different aspects of Deep learning based approaches which are used for image manipulation and will gather some insights about how they work. For this article, we are going to look at the approaches that uses Neural Networks as they sit at the core workings of the algorithm.

Generally we use different variations of the CNN methods to perform NST, but the main idea is the same. To better grasp the underlying functions and their workings, I request you to check out this introductory video on CNN: [Video](https://youtu.be/FmpDIaiMIeA)

![CNN](/assets_for_posts/deeplearning/What_is_neural_style_transfer/cnn.gif)

--------------------------------------------------------------------------------------------------------------------

***⭐ Basic architecture***
&nbsp;
Here are the three different required inputs for the model to perform Image style transfer:

A Content Image – an image to which we want to transfer style to
A Style Image – the style we want to transfer to the content image
An Input Image (generated) – the final blend of content and style image

NST uses Deep Learning models to perform the image transformation. A pre-trained Convolutional Neural Network is used with added loss functions to extract the content of the image and transfer the style from one image to another. When we use CNN, we input the image and the image is outputed as a vector of feature maps. Those features maps focus on the content of the image rather than the small details.

Training a style transfer model is a multi-step process. First, we need to define the style and content images. Then comes, the hyperparameters tuning for style transfer model. and at last we need to define the loss function, optimizer and number of the training loops.

To save an enormous amount of training time, we use a pre-trained feature extractor. Different layers of the CNN help us to extract the different kind of features, some layers learn to extract the shape of a cat or the tyres of teh car, while others layers learn to focus on the minor details for textures. As Style transfer exhibits this by running two images through a pre-trained neural network, using the output at different layers, it compares the similarity between the content of the two images, as the layers producing same kind of output will likely have same kind of content at that furbhised layer.

But the job isn't done yet, the model enables us to compare the similar amount of content and style of the two images, but in fact the model has no role to stylized the image. Styling the image is the job  of the second neural network which is called Transfer Network. Transfer network is an implementation of the image translation network which takes one image as an input and give another image as an output. They use *Encoder-Decoder Architecture* for this purpose.

![fastst](/assets_for_posts/deeplearning/What_is_neural_style_transfer/faststyletransferarch.jpg)

At the start of the training, one or more style images are passed through the pre-trained feature extractor and the outputs from the various style layers are saved for later comparison. Following that, the content images are fed into the model, each given content image is passed through a pre-trained feature extractor as stated earlier. Output is recorded from the different content layers are recorded, and then the content image is passed through the Transfer network which gives stylized image as an output. The stylized image is also passed through the feature extractor and the output from both the content and style layers is saved.

--------------------------------------------------------------------------------------------------------------------

***🚀 But how to change the quality and custom preferences of the stylized image?***

&nbsp;
The quality of the stylized image is controlled by the loss function. The loss function is a function that takes the stylized image and the content image as input and outputs a scalar value. The extracted content features are compared with the content image, and the extracted style image features are compared to the referenced style image. After every iteration, the transfer network is updated based on the results of the loss function. Important to note is that the loss function is not a function of the stylized image, but of the content and style images.

***🔴 NOTE: The pre-defind weights of the feature extractor remains same throughout the operation, as changing the weights of the different terms of the of the loss function can help us to train our models to produce different output images with various combinations of the lighter and heavier stylization.***

Content loss is written as

![cl](/assets_for_posts/deeplearning/What_is_neural_style_transfer/cl.png)

This is based on the intuition that images with similar content will have similar representation in the higher layers of the network. P^l is the representation of the original image and F^l is the representation of the generated image in the feature maps of layer l.

Where as Style loss is

![sl](/assets_for_posts/deeplearning/What_is_neural_style_transfer/sl.png)

Here, A^l is the representation of the original image and G^l is the representation of the generated image in layer l. Nl is the number of feature maps and Ml is the size of the flattened feature map in layer l. wl is the weight given to the style loss of layer l.

So total loss would be

![tl](/assets_for_posts/deeplearning/What_is_neural_style_transfer/tl.png)

So our total loss function basically represents our problem - we need the content of the final image to be similar to the content of the content image and the style of the final image should be similar to the style of the style image.

Two Dogs which are exact copies of one another rotated 180 degrees will have a similar representation in the higher layers of the network. As the style loss will be high compared to the content loss..

![loss](/assets_for_posts/deeplearning/What_is_neural_style_transfer/loss.png)

As stated earlier, Neural style transfer uses a pretrained [Convolution neural network](https://youtu.be/FmpDIaiMIeA). Then to define a loss function which blends two images seamlessly to create visually appealing art. Below image shows the architecture of the model in singular steps.

![styletransfering](/assets_for_posts/deeplearning/What_is_neural_style_transfer/styletransfering.png)

If you are feeling confused about the architecture of the model, then don't worry, the image is just a simple walkthrough for you, I will cover all the important aspects in detail in the next several sections to come.. Bear with me. Till then if you are mesmerized by NST, see this amazing video to implement it on your own: [Video](https://youtu.be/bFeltWvzZpQ).

--------------------------------------------------------------------------------------------------------------------
📕References

- 📑 [Article 1](https://towardsdatascience.com/a-brief-introduction-to-neural-style-transfer-d05d0403901d)
- 📑 [Article 2](https://distill.pub/2017/feature-visualization/)
- 📑 [Article 3](https://heartbeat.comet.ml/a-beginners-guide-to-convolutional-neural-networks-cnn-cf26c5ee17ed)
- 🎥 [Video 1](https://youtu.be/bFeltWvzZpQ)
- 🎥 [Video 2](https://youtu.be/FmpDIaiMIeA)

--------------------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
