import os

thedir = '.'
with open("README.md", mode='w') as f:
    f.write("# Datasets information\n\n")
    for dir in [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]:
        mat = len(tuple(filter(lambda x: 'mat' in x, os.listdir(os.path.join(thedir, dir))))) == 1
        pict = len(tuple(filter(lambda x: 'jpg' in x, os.listdir(os.path.join(thedir, dir)))))
        f.write("*{}\n\tSize:{}\n\tMat:{}\n\n".format(dir, pict, mat))