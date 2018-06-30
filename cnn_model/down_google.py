from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"parking inside,market hall inside,arena inside",
             "limit":100,
             "print_urls":True,
             "type":"photo",
             "format":"jpg",
             "size":"medium"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images