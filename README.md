# AUTOFORM
Use AI to auto format slides based on existing odf slides

## To-do 
* [x] Implement simple NN to start training data
* [ ] Start extracting features and train formatting data 
* [ ] Improve the NN to have multiple layers

## Features
* [x] Implement simple slide loader to get a list of elements already on it (getting a XML string)
* [x] Move to rust repo
* [x] Use zip library to unzip the content  
* [x] Added XML parsing library 
* [x] Attempt to retrieve properties related to svg: position and size elements 
* [x] Manipulate content.xml file to move things around 
* [x] Save it back to odp format 
* [x] Use a randomizing algorithm to style the slides 

## Build

### Windows
 `scoop install gcc`

## References
* How to avoid unwrap (https://dmerej.info/blog/post/killing-unwrap/)
* How to parse ODF file using python as example (https://www.linuxjournal.com/article/9347)
* Deep Neural Networks from scratch in Python (https://towardsdatascience.com/deep-neural-networks-from-scratch-in-python-451f07999373)

## Open Issues
* [ ] Fixed the code to support self-closing tag as well 
