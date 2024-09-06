<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css"> </head>
<body>
    <h1>Gender Classification from Hebrew Handwritten Text Images</h1>
    <h2>Introduction</h2>
    <p>This project aims to develop a Convolutional Neural Network (CNN) model capable of classifying gender (Male/Female) based on Hebrew handwritten text images. The model will be trained and evaluated using the hhd_dataset.</p>
    <h2>Dataset</h2>
    <ul>
        <li><strong>Name:</strong> hhd_dataset</li>
        <li><strong>Description:</strong> A dataset containing Hebrew handwritten images labeled with gender.</li>
        <li><strong>Format:</strong> TIFF</li>
        <li><strong>Size:</strong>
            <ul>
                <li>Train: 614</li>
                <li>Validate: 133</li>
                <li>Test: 147</li>
            </ul>
        </li>
    </ul>
    <h2>Model Architecture</h2>
    <p>This CNN architecture consists of four convolutional layers with ReLU activations, followed by a fully connected layer and a dropout layer. The model uses a combination of convolutional layers, batch norm layers, pooling layers, and dropout layers to extract features from the input images and prevent overfitting.</p>
    <h2>Preprocessing Stage</h2>
    <ul>
        <li><strong>Image Loading and Inversion:</strong>
            <ul>
                <li>All images were loaded as grayscale.</li>
                <li>The colors of the images were inverted to improve visibility.</li>
            </ul>
        </li>
        <li><strong>Standard Preprocessing:</strong>
            <ul>
                <li>Noise reduction techniques like median filtering or Gaussian blurring were applied.</li>
                <li>Contrast adjustments were made to enhance the distinction between foreground and background.</li>
                <li>Pixel values were normalized to a specific range.</li>
            </ul>
        </li>
        <li><strong>Image Patch Extraction:</strong>
            <ul>
                <li>Images were divided into non-overlapping 400x400 pixel patches.</li>
                <li>Patches were allowed to overlap to increase the number of training samples.</li>
                <li>Approximately 200 new sub-images were extracted from each original image.</li>
            </ul>
        </li>
    </ul>
    <h2>Classification Stage</h2>
          <strong>Classification by Majority Voting:</strong>
          <ul>
              <li>Each extracted patch was independently classified using the trained CNN model.</li>
              <li>The final classification for an entire image was determined by the majority vote of its constituent patches.</li>
          </ul>
    </ul>
    <h2>Results</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Train Accuracy</td>
            <td>64%</td>
        </tr>
        <tr>
            <td>Test Accuracy</td>
            <td>59%</td>
        </tr>
        </table>
    <h2>Summary</h2>
    <ul>
      <li>The model's performance can be further enhanced. One area for improvement lies in the preprocessing stage. Five images from the test dataset were classified using majority voting due to potential information deficiencies. This suggests the need for more robust preprocessing techniques or the inclusion of additional training data to address these cases.</li>
      <li>Another avenue for optimization is the fine-tuning of the model's hyperparameters. By carefully adjusting these parameters, we can potentially improve the model's accuracy and generalization capabilities.</li>
    </ul>
    </body>
</html>
