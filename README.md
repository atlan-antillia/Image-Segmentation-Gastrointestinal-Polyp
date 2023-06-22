# Image-Segmentation-Gastrointestinal-Polyp (Updated: 2023/06/23)
<h2>
1 Image-Segmentation-Brain-Tumor
</h2>
<p>
This is an experimental project for Image-Segmentation of Gastrointestinal-Polyp by using
our <a href="https://github.com/atlan-antillia/Tensorflow-Slightly-Flexible-UNet">Tensorflow-Slightly-Flexible-UNet</a> Model.
<br>
</p>
The image dataset used here has been taken from the following web site.
</p>
<pre>
Kvasir-SEG Data (Polyp segmentation & detection)
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>
<p>
See also:
</p>
<pre>
Kvasir-SEG
https://paperswithcode.com/dataset/kvasir-seg
</pre>

<br>
<h2>
2 Prepare dataset
</h2>

<h3>
2.1 Download master dataset
</h3>
 Please download the original dataset from the following link<br>
<pre>
Kvasir-SEG Data (Polyp segmentation & detection)
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>
<b>Kvasir-SEG</b> dataset has the following folder structure.<br>
<pre>
Kvasir-SEG
├─annotated_images
├─bbox
├─images
└─masks
</pre>

<h3>
2.2 Create master dataset
</h3>
We have split Kvasir-SEG dataset to <b>test</b>,<b>train</b> and <b>valid</b> dataset 
by using Python <a href="./projects/GastrointestinalPolyp/generator/create_master_256x256.py">create_master_256x256.py</a> script.
<pre>
GastrointestinalPolyp
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<b>Augmented image samples: GastrointestinalPolyp/train/images</b><br>
<img src="./asset/GastrointestinalPolyp_train_images_sample.png" width="1024" height="auto"><br>
<b>Augmented mask samples: GastrointestinalPolyp/train/mask</b><br>
<img src="./asset/GastrointestinalPolyp_train_masks_sample.png" width="1024" height="auto"><br>

<h2>
3 Train TensorflowUNet Model
</h2>
 We have trained Gastrointestinal Polyp TensorflowUNet Model by using the following
 <b>train_eval_infer.config</b> file. <br>
Please run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetTrainer.py train_eval_infer.config
</pre>
, where train_eval_infer.config is the following.
<pre>
; train_eval_infer.config
; 2023/6/23 antillia.com
; Added BatchNormalization flag to [model section.
; normalization = True

[model]
image_width    = 256
image_height   = 256
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (3,3)
num_layers     = 6
normalization  = True
dilation       = (1,1)
dropout_rate   = 0.08
learning_rate  = 0.001
loss           = "bce_iou_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"

image_datapath = "./GastrointestinalPolyp/train/images/"
mask_datapath  = "./GastrointestinalPolyp/train/masks/"

[eval]
image_datapath = "./GastrointestinalPolyp/valid/images/"
mask_datapath  = "./GastrointestinalPolyp/valid/masks/"

[infer] 
images_dir    = "./mini_test" 
output_dir    = "./mini_test_output"
merged_dir    = "./mini_test_output_merged"

[mask]
blur      = True
binarize  = True
threshold = 60
</pre>

Since <pre>loss = "bce_iou_loss"</pre> and <pre>metrics = ["binary_accuracy"] </pre> are specified 
in <b>train_eval_infer.config</b> file,
<b>bce_iou_loss</b> and <b>binary_accuracy</b> functions are used to compile our model as shown below.
<pre>
    # Read a loss function name from a config file, and eval it.
    # loss = "bce_iou_loss"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names from a config file, and eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
        
    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
</pre>
You can also specify other loss and metrics functions in the config file.<br>
Example: basnet_hybrid_loss(https://arxiv.org/pdf/2101.04704.pdf)<br>
<pre>
loss         = "basnet_hybrid_loss"
metrics      = ["dice_coef", "sensitivity", "specificity"]
</pre>
On detail of these functions, please refer to <a href="./losses.py">losses.py</a><br>, and 
<a href="https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master">Semantic-Segmentation-Loss-Functions (SemSegLoss)</a>.

We have also used Python <a href="./ImageMaskDataset.py">ImageMaskDataset.py</a> script to create
train and test dataset from the <b>GastrointestinalPolyp</b> dataset specified by
<b>image_datapath</b> and <b>mask_datapath </b> parameters in the configratration file.<br>
The training process has just been stopped at epoch 50 by an early-stopping callback as shown below.<br><br>
<img src="./asset/train_console_at_epoch_50_0623.png" width="720" height="auto"><br>
<br>
The <b>val_accuracy</b> is very high as shown below from the beginning of the training.<br>
<b>Train accuracies line graph</b>:<br>
<img src="./asset/train_metrics_50_0623.png" width="720" height="auto"><br>

<br>
The val_loss is also very low as shown below from the beginning of the training.<br>
<b>Train losses line graph</b>:<br>
<img src="./asset/train_losses_50_0623.png" width="720" height="auto"><br>


<h2>
4 Evaluation
</h2>
 We have evaluated prediction accuracy of our Pretrained GastrointestinalPolyp Model by using <b>./GastrointestinalPolyp/test/images/</b> dataset.
Please run the following bat file.<br>
<pre>
>2.evalute.bat
</pre>
, which simply run the following command.<br>
<pre>
>python ../../TensorflowUNetEvaluator.py train_eval_infer.config
</pre>
The evaluation result of this time is the following.<br>
<img src="./asset/evaluate_console_at_epoch_50_0623.png" width="720" height="auto"><br>
<br>

<h2>
5 Inference 
</h2>
We have also tried to infer the segmented region for <b>mini_test</b> dataset, which is a very small dataset including only 
ten images extracted from <b>test</b> dataset,
 by using our Pretrained  GastrointestinalPolyp Model.<br>
<pre>
>3.infer.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetInferencer.py train_eval_infer.config
</pre>

<b>Input images (mini_test) </b><br>
<img src="./asset/mini_test.png" width="1024" height="auto"><br>
<br>

<b>Ground truth mask (mini_test_mask) </b><br>
<img src="./asset/mini_test_mask.png" width="1024" height="auto"><br>
<br>

<b>Inferred images (mini_test_output)</b><br>
Some green tumor regions in the original images of the mini_test dataset above have been detected as shown below.
<img src="./asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<b>Merged inferred images</b><br>
<img src="./asset/mini_test_output_merged.png" width="1024" height="auto"><br><br>

<br>
<!--
-->


<h3>
References
</h3>

<b>1. Kvasir-SEG Data (Polyp segmentation & detection)</b><br>
<pre>
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>

<b>2. Kvasir-SEG: A Segmented Polyp Dataset</b><br>
Debesh Jha, Pia H. Smedsrud, Michael A. Riegler, P˚al Halvorsen,<br>
Thomas de Lange, Dag Johansen, and H˚avard D. Johansen<br>
<pre>
https://arxiv.org/pdf/1911.07069v1.pdf
</pre>

