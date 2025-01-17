# **EqualEyes: Image Caption Generator**

## **Team**
- Shashank Guda (POC)
 https://github.com/gudashashank
- Soundarya Ravi
 https://github.com/soravi18
- Ramya Chowdary Patchala
 https://github.com/ramyacp14 
- Rithika Gurram
https://github.com/Rithika2000
- Vishnu Charugundla
https://github.com/VishnuCharugundla


## **Introduction**

The project aims to push image captioning technology beyond what currently exists. We will combine recent advances in image recognition and language modeling to generate novel descriptive captions that go beyond just naming objects and actions. The goal is to create a system that provides rich, detailed, and natural descriptions of photographs to make them more accessible and meaningful for all. Training on open and varied image datasets will be key to ensuring the system can generalize well and avoid bias. We will also prioritize evaluation methods that measure how well the captions describe the full context of images, not just presence of objects. There are still some challenges ahead, but by focusing on inclusivity and the uniqueness of this multi-modal approach, we hope to create something that moves image captioning technology forward in an impactful way.

The objective of *EqualEyes* is to be helpful for anyone who interacts with images and relies on captions to understand the content better. This includes individuals with visual impairments who use screen readers (especially the color blind people), people browsing social media platforms or news websites, researchers analyzing image datasets, and developers working on applications that involve image recognition and understanding. Ultimately, the goal is to make image captioning more inclusive and accessible for all users, regardless of their abilities or needs.

The primary stakeholders for this project include individuals with visual impairments and educators. Successful implementation of the system will significantly improve accessibility for individuals using screen readers and provides valuable educational resources for teachers and educators.

The working web application takes an image in any format as an input, processes it through our 4th model BLIP Model, a descriptive caption is generated for that image, further it has an audio extension which helps Visually Impaired People to listen to the description of an image through medium of sound. For now we have established for the standard American English.

#### **Outlining Stakeholders**

**1. Individuals with Visual Impairments:**
- Individuals who rely on screen readers to access image content.
- Accessible and descriptive image captions that accurately convey the content and context of images, enabling them to comprehend visual information effectively.

**2. Social Media Users:**
- Users of social media platforms such as Instagram, Twitter, and Facebook.
- Engaging and informative image captions that enhance the browsing experience and facilitate better understanding of shared images.

**3. Teachers and Educators:**
- Professionals involved in educating their peers, including those who train other teachers in the effective use of technology in the classroom. 
- It creates an engaging learning experience, supports early literacy with younger children where it makes learning fun, it also improves the language learning, it can provide additional vocabulary to the peers,highly useful in bilingual or multilingual classroom environments.


## **Literature Review**

Microsoft Research has developed a system that generates detailed captions for real-world images by combining powerful image recognition with tools for generating, ranking and contextualizing captions. The system utilizes a state-of-the-art image recognition model to understand image content. It can also identify celebrities and landmarks to create richer, more informative captions. Testing shows the system outperforms previous ones, especially for images from diverse sources like Instagram. This research represents a major advance in creating accurate, detailed image captions with practical real-world applications. However, challenges remain in ensuring fairness and inclusivity. Strategies like data augmentation and balanced sampling have been proposed to mitigate biases. 

The system may struggle to generalize beyond its training data and battle biases. Its reasoning abilities are also limited, focusing mainly on basic description rather than deeper relationships. Additionally, evaluating caption quality systematically remains difficult. While the approach marks an advance, further innovation around generalization, reasoning, bias mitigation, evaluation methodology and integration into practical applications is needed to fully deliver on the potential of AI-generated image captions. Addressing these challenges could yield additional breakthroughs in creating inclusive, ethical and useful captioning systems.

The previous works so far were generic approaches in captioning an image, ours focuses on bringing forth more descriptive caption in a human level. We are going beyond object recognition, we are utilising the Microsoft Research Files on Object Segmentation, Instances and Captioning together to train the model better. It even focuses on providing enriched language as one of our key stakeholders are educators for early literacy, we focus primarily on addressing the stakeholder needs here to serve as an application for them. We are incorporating inclusivity in the image captioning by making it as an helpful tool for visually impaired people with audio extension for them, also our approach included objects which are highly used in the sentence formation of educational resources to make it as a keynote in Learing Content Management System applications.

The discussion around improving inclusiveness, mitigating bias, and ensuring equitable performance of image captioning systems across different demographics relates most directly to Section 1.4 of the ACM Code of Ethics and Professional Responsibilities. Specifically:

***“Be fair and take action not to discriminate.”***


## **Data & Methods**

## **Data**

### **Data 1**
For this project, We have utilized the 2017 data from the COCO dataset ([source](https://cocodataset.org/#download)). The COCO dataset is a widely used benchmark for image captioning tasks due to its large scale and diversity.

The dataset consists of images across various categories, each annotated with detailed information such as object detection, segmentation masks, keypoints, etc.

#### **Validation of Dataset Credibility**

The COCO dataset is widely recognized and extensively used in the computer vision community. It has been meticulously curated, and its annotations are of high quality. Additionally, the dataset has been used in numerous research papers and competitions, further validating its credibility.

#### **Diving Deep Into Data**

The COCO 2017 Dataset has around 118k images in train set and 40k in the test set. From that for our models we have utilized around 56k train and 14k validation set and tested random test sets in the predicted column. Each Image has 5 annotation[['captions']], we trained with all of the 5 captions.

#### **Files**
 
- Images
 1. train 2017 
 2. test 2017
 3. val 2017

- Meta Data on Images
  Under the Annotations folder:
1. Captions_train2017
2. Captions_val2017
3. Instances_train2017
4. Instances_val2017
5. Person_keypoints_train2017
6. Person_keypoints_val2017

#### **Explaining Key Notes**

There are 1.5 million Object Instances, 12 super categories and 80 sub categories, also it has 91 stuff categories and there are 250k people with people key points which is annotated with 17 categories in Person_Keypoints.

The Caption JSON file has the image id with the Caption  annotation which has been merged as a dataframe for each image to train the model, further few variables from the Instances and the Person Key points has been fetched to train using Object segmentation and stuff segmentation. 


The segmentation, bbox, category_id, person_category_id, super_category which explains the location of each object, stuff category,person category. 


![Alt Text](Images/Object_Instances.jpg)

![Alt Text](Images/Super_Sub_ID.jpg)

## Data 2: ImageNet

ImageNet ([source](https://www.image-net.org/download.php)) is a large image dataset specifically designed for visual object recognition research. Here's a summary of its key aspects relevant to your project:

#### **Validation of Dataset Credibility**

ImageNet is a very widely known dataset in computer vision research, acknowledged and utilized across academic and industrial domains. Its credibility is rooted in several key factors. Firstly, ImageNet undergoes meticulous curation, ensuring the quality and consistency of its data. Moreover, it serves as the foundation for the influential ImageNet Large Scale Visual Recognition Challenge (ILSVRC), a benchmark competition that significantly propelled the advancement of object detection and image classification algorithms. Additionally, ImageNet's transparency and accessibility contribute to its credibility, being publicly available for researchers to access, analyze, and contribute to, thereby fostering open collaboration and validation in the field of computer vision.

#### **Files**
- **Size:** 1.2 million training images (significantly larger than your COCO dataset).
- **Diversity:** Contains roughly 1,000 object classes, encompassing a wider range of categories than COCO.
- **Structure:** Organized based on the WordNet hierarchy, aiding model understanding of relationships between objects.
- **Annotations:** Image-level annotations indicating object presence (less detailed than COCO's bounding boxes).

## **Methods**

### **Preprocessing**

#### **Text Preprocessing:**

The captions extracted from the COCO dataset underwent several preprocessing steps to prepare them for training. These steps ensure consistency, remove unnecessary characters, and tokenize the text appropriately. Here's an outline of the preprocessing techniques applied:

1. **Lowercasing:** All text was converted to lowercase to ensure consistency in the data.

2. **Removing Non-Alphanumeric Characters:** Non-alphanumeric characters and non-whitespace characters were removed from the captions. These characters typically do not significantly affect the meaning of the text but may introduce noise.

3. **Whitespace Normalization:** Multiple consecutive whitespace characters were replaced with a single space. This normalization step ensures uniformity in text formatting.

4. **Adding Special Tokens:** Special tokens, such as `[start]` at the beginning and `[end]` at the end of each caption, were added to indicate the start and end of sequences. These tokens help the model learn the sequence generation task effectively.

5. **Tokenization:**
   - For tokenization, a TextVectorization layer was initialized to tokenize the captions. The following configurations were applied to the TextVectorization layer:
     - Maximum Sequence Length: Captions were limited to a maximum length of 40 tokens to manage computational complexity.
     - Vocabulary Size: The vocabulary size was set to 15,000 to balance model complexity and coverage of words in the dataset.
     - Output Sequence Length: Sequences were padded or truncated to have a length of 40 tokens, ensuring uniformity in input data format for the model.

#### **Image Preprocessing:**

1. **Decoding and Resizing:** Decode images into JPEG format and resize them to 299 x 299 pixels.
2. **Preprocessing Function:** Utilize a function to preprocess images by decoding them into JPEG format, resizing them, and applying InceptionV3 preprocessing.
3. **Create TensorFlow Dataset:** Pair each image file path with its corresponding caption using `tf.data.Dataset.from_tensor_slices`.
4. **Map Preprocessing Function:** Apply the preprocessing function to each pair in parallel using `map`.
5. **Shuffle and Batch:** Shuffle the dataset and batch it for training.
6. **Image Augmentation:** Define a sequential model for image augmentation, including random horizontal flipping, rotation, and contrast adjustment.

### **Modeling**

#### **Model 1: Basic CNN-RNN model**

Our modeling architecture consists of two main components: an image encoder and a text decoder.

1. **CNN Encoder Function:**
   - Defines a function creating an InceptionV3-based CNN encoder model.
   - Loads the **InceptionV3** model pre-trained on ImageNet without the top layer.
   - Reshapes output to a 2D shape, creating a new model for image feature extraction.

2. **Custom Transformer Encoder Layer:**
   - Defines a layer with multi-head self-attention and feed-forward neural network.
   - Includes layer normalization after both mechanisms.
   - Forward pass normalizes input, passes through feed-forward network and self-attention, and normalizes output.

3. **Custom Embedding Layer:**
   - Defines a layer for computing embeddings in a Transformer model.
   - Comprises token and positional embedding layers, allowing learning of token-based and position-based representations.

4. **Custom Transformer Decoder Layer:**
   - Defines a layer with multi-head self-attention, layer normalization, feed-forward networks, and dropout.
   - Includes method for generating causal attention mask.
   - Forward pass involves self-attention mechanisms, normalization, feed-forward networks, and output generation.

5. **Image Captioning Keras Model:**
   - Integrates CNN model for image feature extraction, Transformer encoder for processing, and Transformer decoder for caption generation.
   - Includes methods for loss computation, custom training, testing, and accessing metrics.

6. **Image Captioning Model Instance Setup:**
   - Creates an instance of the ImageCaptioningModel class.
   - Sets up CNN model, Transformer encoder, Transformer decoder, and optional image augmentation layer.

7. **Model Training Setup:**
   - Compiles the image captioning model with Adam optimizer and sparse categorical cross-entropy loss.
   - Includes early stopping callback based on validation loss.

8. **Model Training Process:**
   - Trains the image captioning model using provided dataset for specified epochs.
   - Validates model performance with a validation dataset.
   - Applies early stopping to prevent overfitting.

9. **Utility Functions:**
   - load_image_from_path: Loads and preprocesses an image from a file path.
   - generate_caption: Generates a caption for an image using the trained model, optionally adding noise.

10. **Caption Generation and Inspection:**
    - Selects a random image from the dataset, generates a caption, and prints it.
    - Displays the selected image for visual inspection of caption quality.
    - Generates captions token by token using a softmax layer at the output.

**InceptionV3:**

**Pre-trained on ImageNet:**
InceptionV3 is pre-trained on the ImageNet dataset, which contains millions of labeled images across thousands of categories. Pre-training on ImageNet allows the model to learn general features from images, such as edges, textures, and object shapes.

**Architecture:**
InceptionV3 is a convolutional neural network (CNN) architecture developed by Google. It is a deep learning model consisting of multiple layers of convolutional and pooling operations.

**Layers:**
1. **Input Layer:** The model starts with an input layer (InputLayer).

2. **Convolutional Layers:** Following the input layer, the model contains several convolutional layers (Conv2D).

3. **Activation Layers:** After each convolutional layer, an activation layer (Activation) applies a non-linear activation function to introduce non-linearity into the model.

4. **Pooling Layers:** Pooling layers (MaxPooling2D and AveragePooling2D) are interspersed with convolutional layers to reduce spatial dimensions and control overfitting.

5. **Concatenation Layers:** At certain points in the model, the feature maps from multiple branches are concatenated (Concatenate) to combine different levels of abstraction.

6. **Output Layer:** The final layers of the model are responsible for producing the model's output.

#### **Model 2: CNN-RNN with Hyper-parameter tuning model**

Modifications were implemented in the CNN encoder, including adjustments to the maximum sequence length and the addition of a learning rate parameter (learning_rate=0.0001) in the Adam optimizer.

1. **CNN Encoder Modification:** The code for the Convolutional Neural Network (CNN) encoder has been altered.

2. **Maximum Length Adjustment:** The maximum length parameter (MAX_LENGTH) has been updated to 40, indicating that sequences longer than 40 tokens will be truncated or padded to fit this length.

3. **Vocabulary Size Specification:** The vocabulary size (VOCABULARY_SIZE) has been set to 15,000, determining the number of unique words considered in the model's training.

4. **Batch Size Definition:** The batch size (BATCH_SIZE) for training has been specified as 64, indicating the number of samples processed in one iteration.

5. **Buffer Size for Shuffling:** The buffer size (BUFFER_SIZE) for shuffling the dataset has been set to 1000, determining the number of elements from which the next element is randomly chosen.

6. **Embedding Dimension Assignment:** The embedding dimension (EMBEDDING_DIM) has been assigned a value of 1024, specifying the size of the dense embedding vector for each token.

7. **Units in Neural Network Layers:** The number of units (UNITS) in the neural network layers has been set to 1024, determining the dimensionality of the output space.

8. **Training Epochs:** The number of training epochs (EPOCHS) has been defined as 10, indicating the number of times the entire dataset will be passed forward and backward through the neural network.

Additional Changes:
- **Learning Rate in Adam Optimizer:** A learning rate of 0.0001 has been added to the Adam optimizer, which controls the step size at each iteration while moving toward a minimum of the loss function during training.

#### **Model 3: Vision Transformer: ViT-GPT2**

The Third Model that we used here is a vision transformer-based architecture combined with GPT-2 for image captioning. It takes an image as input, processes it through a vision encoder, and generates a descriptive caption using a decoder based on GPT-2 architecture.

Vision transformers are a type of transformers that perform visual-related tasks that include images. They are a transformer that also use attention mechanisms to find the relationships between input images.

The three pre-trained models are:

- VisionEncoderDecoderModel: This helps in carrying out an image-to-text generation with any pre-trained vision model using a Transformers (as the encoder) such as ViT (which we used here) or BEiT kind of models which uses self-supervised pre-training of Vision Transformers (ViTs) to outperform supervised pre-training alongside any pre-trained language model as the decoder such as GPT2 (which we are also using here). So, in this approach, we employ VisionEncoderDecoder as an application for image captioning using it to encode the image and later use a language model to generate the captions.

- GPT2TokenizerFast: This creates a GPT-2 tokenizer using the Hugging Face tokenizers library. We load the tokenizers library to the transformers. The tokenizer has been trained already to handle all the feats we require for captioning.

- ViTImageProcessor: Lastly is the ViTImageProcessor. It helps to construct a ViT image processor.

ViTs are deep learning models that process sequential input data and reduce training times. Using the pre-trained models VisionEncoderDecoderModel, GPT2TokenizerFast, and ViTImageProcessor, provided an easy way of building without building from scratch. They also have the ability to outperform supervised pre-training and are suitable for image captioning.

### **Model 4: BLIP: Bootstrapping Language-Image Pre-training**

The Fourth Model that we used here is BLIP: Bootstrapping Language-Image Pre-training for image captioning, from model and processor initialization to caption generation for input images.

BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. BLIP also demonstrates strong generalization ability when directly transferred to video language tasks in a zero-shot manner.

BLIP pre-trains a multimodal mixture of encoder-decoder model using a dataset bootstrapped from large-scale noisy image-text pairs by injecting diverse synthetic captions and removing noisy captions.

This is multimodal mixture of encoder-decoder (MED), a multi-task model which can operate in one of the three functionalities:

1. **Unimodal encoder**, which separately encodes image and text. The text encoder is the same as BERT (Devlin et al.,2019), where a `[CLS]` token is appended to the beginning of the text input to summarize the sentence.

2. **Image-grounded text encoder**, which injects visual information by inserting one additional cross-attention (CA) layer between the self-attention (SA) layer and the feed forward network (FFN) for each transformer block of the text encoder. A task-specific `[Encode]` token is appended to the text, and the output embedding of `[Encode]` is used as the multimodal representation of the image-text pair.

3. **Image-grounded text decoder**, which replaces the bidirectional self-attention layers in the image-grounded text encoder with causal self-attention layers. A `[Decode]` token is used to signal the beginning of a sequence, and an end-of-sequence token is used to signal its end.

**Fine-tuning Parameters:**
- Definition of parameters such as batch size, learning rate, and number of training epochs.

**Model Optimization and Fine-Tuning:**

1. **Optimizer Initialization:**
   - Initialization of an AdamW optimizer with parameters from a pre-trained BLIP model.

2. **Fine-Tuning Loop Setup:**
   - Determination of computation device (GPU or CPU).
   - Transfer of the pre-trained BLIP model to the chosen device.
   - Setting the model to training mode.

3. **Training Loop for Fine-Tuning:**
   - **Epoch Iteration:**
     - Iteration over each epoch, tracking epoch number and accumulating epoch loss.
   - **Batch Processing:**
     - Processing of batches of training data within each epoch.
     - Extraction of image paths and corresponding captions.
     - Generation of captions for each image.
     - Tokenization of captions and images using the BLIP processor.
     - Forward pass through the model.
   - **Loss Computation and Backpropagation:**
     - Computation of loss between predicted and ground-truth captions.
     - Accumulation and averaging of loss over batches.
     - Backpropagation to update model parameters using the optimizer.
   - **Epoch Completion:**
     - Computation and printing of average loss for the epoch.

**Image Caption Generation:**
- Loading of BLIP processor and pre-trained model trained on the COCO dataset.
- `generate_caption` function: Takes an image, processor, and model, processes the image, generates a caption using the model, and returns the caption.
- `generate_captions` function: Wrapper for `generate_caption`, taking an image as input and returning the generated caption using predefined processor and model.
- Base path for COCO dataset and model loading provided.


## **Results**
So far we have done 4 models, first model was the basic **CNN encoder using Inception V3** for image encoding and RNN decoder for text decoding, and we have seen a loss of **2.61.**

Our **model 2** is the optimized version of model 1 which is tuned using hyperparameters like learning rate, Adam optimizer, vocabulary size, number of epoch(20), batch size and we have managed to decrease the loss to **2.31.**

Our third model is a **Vision Transformer GPT-2** model, upon fine tuning it and increasing the epoch size to 25, we have reduced the loss to **0.376**

Our final model is a **BLIP** model, upon fine tuning it and using the epoch size 25, we have further reduced the loss to **0.0062.**

The application which we have developed is built on **BLIP: Bootstrapping Language Image Pre-training** for image captioning. It gave lesser loss compared to the other three models, moreover it has a filter to remove noise from website data making it a better fit for the application.

Attaching a sample Image below from the application: 
![Alt Text](Images/Results_Section.jpg)


### **BLEU Score**

The BLEU (Bilingual Evaluation Understudy) score is a metric used to evaluate the quality of machine-generated text, such as translations or captions. It measures how closely the generated text matches human-written reference text, based on the overlap of n-grams (sequences of n words) between the two. A higher BLEU score indicates a closer resemblance between the generated and reference texts, suggesting better quality. The average BLEU score of **0.72** indicates that, on average, the generated captions closely match the reference captions by approximately **72%.**

![Alt Text](Images/BLEU.jpg)


## **Discussion**

We have done 4 models which has been explained above in the methods section. The first one is the CNN encoder like inception V3 for image encoding and the RNN decoder for the text decoding, we used a mid level batch and epoch size given the CPU/GPU constraint, was able to increase the accuracy and decrease both training and validation loss. Further we tweaked the hyper-parameters and have done the second modeling using the same CNN/RNN, here we tuned the learning rate, utilized the adam optimizer, and also increased the vocabulary size to frame for descriptive captions, it had a lesser loss from 2.61 to 2.31, and the captions were able to frame better sentences. 


Here the model 1 had a training time of ~14 hours, with additional parameter optimization second model took around 25 hours in the local system but it gave lesser training and validation loss.


Further we have utilized a pre-trained model Vision Encoder Decoder model, here we used the Vision architecture to find the relationship between the images and GPT2 decoder to write the caption. It has  reduced the training time to 8 hours and we were able to increase the epochs size, model is already trained on image net data, we fine tuned it to fit the coco-data file and annotation set. The loss drastically reduced to 0.378. We did a fourth model further using BLIP, which further reduced the loss to 0.0062.



We are quite satisfied with where we are in the results at the moment, we are still fine tuning things to make it work better as we want to capture every instance an image can display. We are stuck with the computational resources which has been explained detailed in the Limitation section below. There are 80 sub categories of the vast data set but its not very much enough to cover everything in the world.

The task is GPU bound, but everything we have done so far is using a high computational CPU with 48 cores, so there was a time constraint and idling out issue which restricted in finding more bbox(Boundary Boxes for Object Segmentation) which has been displayed in the image below.


![Alt Text](Images/cpu_idling.jpg)

Explanation: Here our model works best with objects we trained it on, for example few animals, fruits, food categories..etc, here the below image has a caption quoting it as a clock tower instead of captioning it as a Eiffel tower, we trained a model how to identify an object as a clock/clock tower as its one of the object instance in coco2017 dataset. We wanted to use more object and stuff categories, but even here due to disk limitation and CPU idling out we had to train on less than half of the data.


![Alt Text](Images/EiffelTower_ClockTower.jpg)


#### **Addressing the Stakeholder Needs**

Our main primary stakeholders are the people with Visual Impairments, the web-application we created aids them in understanding the images better, given they are traveling to a new place or want to know what is happening around without someone aiding them, this application works better for them as it even has text to audio extension and descriptive caption which captures lot of things happening right there. 

People who reach a malignant stage of diabetic retinopathy lose more than half their sight and cannot distinguish objects properly. With this application, they don't need an additional person to visualize something.

Our second stakeholders are the social media users, we have tested it with random images from the search engines, its able to caption those images, people can utilize these to explain an image they are posting in a generic way, for example, flower garden in middle of a city park, its one of the caption our web application generated when we tested out. It makes browsing engaging, with more classified training we could even utilize it to identify plant species to very specific bird names.
![Alt Text](Images/Garden.jpg)


Our third stakeholders are the teachers and educators, this could be next LCMS(Learning Content Management System) in schools, instead of pages of notes and efforts, they could support early literacy with advanced vocabulary from the captions generated, a child would be intrigued to see different grammatical way of explaining a picture, this would increase the language development at the earlier age. 

For further improvement, we even trained our models with an extensive vocabulary from five different captions.


## **Limitations**

Our main Limitation was the computational resource allocation, it happened at the last moment, it took some learning curve to set it up all together and to get 40GB of image data in, our models are GPU bound, but the GPU was not initialized with the tensorflow version.

We tried two different approaches to get this working faster as each model took from 8-25 hours, we tried creating a virtual machine with high end CPU and also we were able to get the data in using the Azure CLI integration, but again it did not have any GPU access in it, so we switched to the Azure ML Instance which was allocated.

In this second approach, we got the data in through azure blob storage, again it had some issues with unzipping and mounting inside the notebook as the process expected it to be in same working directory, so we uploaded and unzipped right inside the path [['Users/soravi/COCO_Extracted']]. Here the main issue was, even the multi core CPU with 48 cores started idling out after an hour or two given the model ran for 8+ hours, it started timing out and the disk utilization went to zero.

We tried multiple ways to initialize the GPU, but the tensorflow version inside the azure ml instance had a compatibility issue with the GPU setting, the tensorflow version 2.11.0 is not compatible for the tensorflow-gpu installation. [[attached- gpusetup notebook for the results]], we tried to downgrade it to archive version but it needed downgrading of numpy, flatbusters and few other libraries and it started conflicting with the code. CUDA drivers were installed even to address this but given the nick of time, we had to switch to parallel running of 4 models locally in our systems to get results.

For the primary stakeholder of visually impaired people, we wanted to make something which can take an image, puts a bbox(boundary box) over every objects available there and explain each corner of image to give them a visual image in the head rather than one caption which gives a generic overall caption. Given the training restriction with lesser epoch and batch size, we couldn't achieve much in this area.

We collected a lot of object instances which describes more generic objects for higher educational stakeholder but we only trained on COCO 80 object categories as of now. 

Given a GPU which reduces training time, we could have experimented integrating multi-model approach more and could have increased vocabulary size to have more descriptive explanation of what is happening in the image.



## **Future Works**
We are still trying to work on the GPU issue, and we also raised a ticket with Microsoft regarding this. They have suggested a few ways  and we are working on them to get the GPU detected as a physical device.

Our other future scope is to include more advanced vocabulary training as well as to incorporate at least 300 object categories. We want to increase the scope for stakeholders, and for our current users, we want to add more languages such as Italian, Spanish, and Dutch. Just translating English to the desired language through APIs wasn't perfect, so caption training with regional language annotation files is the next short-term goal.

For the long term, given the resources, we wanted to publish this as an application with different screens and different models trained to address stakeholders. For instance, a page for botanists which has a BLIP or Vision Model (as they are high performing ones) with a wide diverse dataset of plant species. This could describe what species are present in an image or what it is similar to. It would be highly helpful for students in graduate schools to have an interactive learning experience.

