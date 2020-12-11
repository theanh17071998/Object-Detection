from make_datapath import make_datapath_list
from lib import *
from extract_inform_annotation import Anno_xml
from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
	PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
	ToPercentCoords, Resize, SubtractMeans

class DataTransform():
	def __init__(self, input_size, color_mean):
		self.data_transform = {
			"train": Compose([
				ConvertFromInts(), # convert image from int to float32
				ToAbsoluteCoords(), # back annotation to normal type
				PhotometricDistort(), # change color by random
				Expand(color_mean),
				RandomSampleCrop(), # random crop image
				RandomMirror(), 
				ToPercentCoords(), # chuan hoa annotation data ve dang [0-1]
				Resize(input_size),
				SubtractMeans(color_mean),
			]), 
			"val": Compose([
				ConvertFromInts(), # convert image from int to float32
				Resize(input_size),
				SubtractMeans(color_mean),
			])
		}

	def __call__(self, image, phase, boxes, labels):
		return self.data_transform[phase](image, boxes, labels)

if __name__ == "__main__":
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
				"sheep", "sofa", "train", "tvmonitor"]

	# prepare train, valid, annotation list
	root_path = "./data/VOCdevkit/VOC2012/"
	train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
	
	# read image
	img_file_path = train_img_list[0]
	image = cv2.imread(img_file_path)
	height, width, channels = image.shape

	# annotation information
	trans_anno = Anno_xml(classes)
	anno_info_list = trans_anno(train_annotation_list[0], width, height)

	color_mean = (104, 117, 123)
	input_size = 300
	transform = DataTransform(input_size, color_mean)

	# transform img
	phase = "train"
	train_img_transformed, boxes, labels = transform(image, phase, anno_info_list[:, :4], anno_info_list[:, 4])

	phase = "val"
	val_img_transformed, boxes, labels = transform(image, phase, anno_info_list[:, :4], anno_info_list[:, 4])

	cv2.imshow("image", image)
	cv2.imshow("train", train_img_transformed)
	cv2.imshow("val", val_img_transformed)
	cv2.waitKey()
