# Nutritionist

<img src=".assets/img/Pie.png" width="600" /> 
<img src=".assets/img/Overview.png" width="600" /> 
# :memo: Description

This project is about a dashboard app built in Dash about the nutritional value of food. It let's the user load, modify and create recipes. The app displays the nutritional value of a serving in comparison to the daily limits.
The dashboard is comprises of the left input panel to modify the input data. The right side is split in two tabs, which gives a quick overview over macro nutrients. The detail view tab allows to take a deep dive on how each ingredients contribution.  
The project is deployed using Render is **currently live** at: https://nutritionist-yrrn.onrender.com/
**Note:** It might take a while to load the page, due to the free tier option of Render I used.

# :page_facing_up: Script overview
| Script       | Function                                               |
| ------------ | ------------------------------------------------------ |
| app.py       | Main script defining layout and app behavior           |
| formats.py   | Contains formats used in app                           |
| transform.py | Contains function to perform data manipulation         |
# :file_folder: Data

The food database downloaded and reduced in size from https://www.myfooddata.com, which sourced the the data from the [U.S. department of Agriculture ](https://fdc.nal.usda.gov/).
