# Semantic Segmentation For Images containing Cell Neuclei.

## Project Description
### i.	What this project does?
This project is to create a model for semantic segmentation for images containing cell neuclei by building a U-Net model.
### ii.	Any challenges that was faced and how I solved them?
- Data preprocessing was done by expanding the mask dimension to include the channel axis. Then, the mask value was converted into just 0 and 1. Lastly, the images pixel value was normalized.
- For the model development process, transfer learning was applied by using a pretrained model as the feature extractor.
Then, the process proceeded to build my own upsampling path.
### iii.	Some challenges / features you hope to implement?
I hope I could explore multi-scale or pyramid features. Integrating multi-scale or pyramid features into the U-Net architecture can enhance the model's ability to capture both local and global contextual information. Techniques such as skip connections between different resolution layers or using feature pyramids can be explored in the future.
## How to install and run the project 
Here's a step-by-step guide on how to install and run this project:

1. Install Python: Ensure that Python is installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org/) and follow the installation instructions specific to your operating system.

2. Clone the repository: Go to the GitHub repository where your .py file is located. Click on the "Code" button and select "Download ZIP" to download the project as a ZIP file. Extract the contents of the ZIP file to a location on your computer.

3. Set up a virtual environment (optional): It is recommended to set up a virtual environment to keep the project dependencies isolated. Open a terminal or command prompt, navigate to the project directory, and create a virtual environment by running the following command: python -m venv myenv

   Then, activate the virtual environment:

   If you're using Windows: myenv\Scripts\activate

   If you're using macOS/Linux: source myenv/bin/activate

4. Install dependencies: In the terminal or command prompt, navigate to the project directory (where the requirements.txt file is located). Install the project dependencies by running the following command: pip install -r requirements.txt

   This will install all the necessary libraries and packages required by the project.

5. Run the .py file: Once the dependencies are installed, you can run the .py file from the command line. In the terminal or command prompt, navigate to the project directory and run the following command: python your_file.py

   Now, you're done! The project should now run, and you should see the output or any other specified behavior defined in your .py file.

## Output of this project
#### i. Model Accuracy:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_cell_nuclei_segmentation/main/Model%20Accuracy.PNG)

#### ii. Model Architecture:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_cell_nuclei_segmentation/main/U_net%20Model%20Architecture.PNG)
 
#### iii. Training Process Accuracy:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_cell_nuclei_segmentation/main/Training%20Process_Acuuracy.PNG)
 
#### iv. Training Process Loss:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_cell_nuclei_segmentation/main/Training%20Process_Loss.PNG)
 
#### v. Model Deployment (Predictions):

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_cell_nuclei_segmentation/main/Model%20Deployment.PNG)

## Source of datasets : 
https://www.kaggle.com/competitions/data-science-bowl-2018/overview

