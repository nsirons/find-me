from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"sport gym,arena inside,room,hall,basketball court,indoors hall,indoors arena,storage,museum inside,swimming pool","limit":100,
             "print_urls":True, "type":"photo", "chromedriver":"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedrive.exe"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images