import os

catnames = []

def dataloader(filepath):
    catname_txt = filepath + '/ImageSets/2017/val.txt'

    global catnames
    catnames = open(catname_txt).readlines()

    annotation_all = []
    jpeg_all = []

    for catname in catnames:
        anno_path = os.path.join(filepath, 'Annotations/480p/' + catname.strip())
        cat_annos = [os.path.join(anno_path,file) for file in sorted(os.listdir(anno_path))]
        annotation_all.append(cat_annos)

        jpeg_path = os.path.join(filepath, 'JPEGImages/480p/' + catname.strip())
        cat_jpegs = [os.path.join(jpeg_path, file) for file in sorted(os.listdir(jpeg_path))]
        jpeg_all.append(cat_jpegs)

    return annotation_all, jpeg_all


def dataloader_jpeg(filepath):
    catname_txt = filepath + '/ImageSets/2017/train.txt'
    catnames = open(catname_txt).readlines()

    refs_train = []
    frame_interval = 1
    for catname in catnames:
        jpeg_path = 'JPEGImages/480p/' + catname.strip()
        cat_jpegs = [os.path.join(jpeg_path, file) for file in sorted(os.listdir(os.path.join(filepath, jpeg_path)))]

        refs_images = []
        n_frames = len(cat_jpegs)
        for i in range(1, n_frames):
            img_batch = [cat_jpegs[0], cat_jpegs[i-1], cat_jpegs[i]]
            refs_images.append(img_batch)
        refs_train.extend(refs_images)

    return refs_train