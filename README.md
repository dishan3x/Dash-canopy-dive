# Dash-canopy-dive
A simple web app can identify canopy feature running through onnx model. 

This application used  onnxruntime which only works for python 3.7 > version.

I tried playing with the windows version of it. But i was unable to use run it due to python errors. 

I was able to run onnxruntime on anaconda environment. It wasnt successful due to path limitation. 

The best way to try this out is using a ubuntu environment. 

pip install dash<br>
pip install onnxruntime<br> 
pip install dash-bootstrap-components

requirment file will be added in near future.

Residue quantifying and analysis application program based on deep neural network models 

This research is based on quantifying the soil residue, soil, and canopy by processing/analyzing image data.  Deep learning is a modern technique in image processing and data analysis which produced promising results that help in many fields.  This application uses a deep learning method called semantic segmentation, to analyze the pixel data in an image.  This process identifies residue in an image and calculates the percentage of each category based on the pixel.   

Solid residues are stems and stalks that remain on soil from previous crops. Residue cover is utilized in farming techniques such as no-till farming. This farming technique uses residue as a cover to the soil layer.  It acts as a barrier to the soil by deflecting energy from the raindrops that can wash away the soil particles and the nutrients in the soil.[1].  

The drawback of having residue till farming is soil loses its ability to control the weeds. The tilled farming processes diminish weed content on the soil. But no-till farming carries untreated weeds and plant diseases which will have to control by herbicides. But due to its benefits, no-till farming has become popular among farmers [2].  

Having a residue cover over the soil will help to prevent soil erosion and wind erosion. It also increases the yield retaining higher water infiltration and nutrient levels in the soil.[2]  It also helps to sustain the moisture in the soil acting as a shield from the sun and preserve water level from drying out from the sun.   

There are many ways that agronomists calculate the residue on soil. Here are a few methods in  

Used in counting soil residue. 

Line transect 

Line transect is a method that uses every 50 – 100-foot tape and counts the residue particles intersect with tape across the sample area on the field. Method counts the number of times the residue intersects with tape for each 1-foot interval. The same counting process is repeated having consistent jumps around a sample area and get the cumulative occurrence as a percentage of the canopy in the area. [3] 

 

Meter stick  

This method takes measurements by throwing a stick into the air and calculating the stick randomly and taking the measurements where it lands and counts residue occurrences alone the stick. 

 

Photo comparison  
In the photo comparison method, they use images predefined residue percentages which already observed and do a comparison between the soil residue and the eye 

 

 

 

Research shows that after cultivating crops the acceptance level for residue level should be maintained around 30 percent [2]. Therefore, there is a necessity for the accuracy of reading the residue cover. 

For example, line transect uses 5 measurements in one sample area and can achieve accuracy of (+) or (-) 15 percent of the mean, and the three measurements will be accounted for accuracy of (+) or (-) 32 percent of the mean. This indicates that to get more accuracy it would need a large number of reading from one sample range. This could be a hard process and time-consuming process to achieve higher accuracy.[4] 

 

The purpose of developing this program application is to improve user efficiency to calculate the residue percentage. It can distinguish the difference between stubble and the color difference just by using one image. It will have the ability to identify the different shapes of residue on the ground which vary from properties such as color and shape.  

 

References  

[1]“Healthy Soil is Covered Soil | USDA.” [Online]. Available: https://www.usda.gov/media/blog/2015/07/16/healthy-soil-covered-soil. [Accessed: 16-Jul-2020]. 

[2]“Methods for measuring crop residue | Integrated Crop Management.” [Online]. Available: https://crops.extension.iastate.edu/encyclopedia/methods-measuring-crop-residue. [Accessed: 16-Jul-2020]. 

[3]E. C. Dickey and J. Havlin, “Estimating Crop Residue-Using Residue to Help Control Wind and Estimating Crop Residue-Using Residue to Help Control Wind and Water Erosion Water Erosion.” 

[4]“(No Title).” [Online]. Available: https://www.blogs.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs142p2_022074.pdf. [Accessed: 16-Jul-2020]. 

Coud not upload the unet model due to file size restrictions in github. 
[link](https://git-lfs.github.com/)