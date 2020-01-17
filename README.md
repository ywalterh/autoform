# AUTOFORM
Use AI to auto format slides based on existing odf slides

## To-do 
* [ ] Implement simple NN to start training data

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

## Open Issues
* [ ] Fixed the code to support self-closing tag as well 
