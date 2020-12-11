# Build module of SSD model
from lib import *
from default_boxes import DefBox
from l2_norm import L2Norm

def create_vgg():
	layers = []
	in_channels = 3
	cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

	for cfg in cfgs:
		if cfg == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		elif cfg == 'MC':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
		else:
			cov2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)

			layers += [cov2d, nn.ReLU(inplace=True)] # xac dinh co luu gia tri dau vao o memory hay khong #True la khong luu
			in_channels = cfg

	pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
	conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
	conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

	layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

	return nn.ModuleList(layers)

def create_extras():
	layers = []
	in_channels = 1024
	cfgs = [256, 512, 128, 256, 128, 256, 128, 256]
	layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
	layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]
	layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
	layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]
	layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
	layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
	layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
	layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]

	return nn.ModuleList(layers)

def create_loc_conf(num_classes=21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
	loc_layers = []
	conf_layers = []

	# source1
	# loc
	loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

	# source2
	# loc
	loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]

	# source3
	# loc
	loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]

	# source4
	# loc
	loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

	# source5
	# loc
	loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]

	# source6
	# loc
	loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
	#conf
	conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]

	return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
	"num_classes": 21, # VOC data
	"input_size": 300, # SSD300
	"bbox_aspect_num": [4, 6, 6, 6, 4, 4],
	"feature_maps": [38, 19, 10, 5, 3, 1],
	"steps": [8, 16, 32, 64, 100, 300],
	"min_size": [30, 60, 111, 162, 213, 264],
	"max_size": [60, 111, 162, 213, 264, 315],
	"aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
}

class SSD(nn.Module):
	def __init__(self, phase, cfg):
		super(SSD, self).__init__()
		self.phase = phase
		self.num_classes = cfg["num_classes"]

		# create main module
		self.vgg = create_vgg()
		self.extras = create_extras()
		self.loc, self.conf = create_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
		self.L2Norm = L2Norm()
		# default box
		dbox = DefBox(cfg)
		self.dbox_list = dbox.create_defbox()

		if phase == "inference":
			self.detect = Detect()

	def forward(self, x):
		sources = []
		loc = []
		conf = []

		for k in range(23):
			x = self.vgg[k](x)
		#source1
		source1 = self.L2Norm(x)
		sources.append(source1)

		for k in range(23, len(self.vgg)):
			x = self.vgg[k](x)
		sources.append(x) # source2

		# source3-6
		for k, v in enumerate(self.extras):
			x = F.relu(v(x), inplace=True)
			if k%2 == 1:
				sources.append(x)
		for (x, l, c) in zip(sources, self.loc, self.conf):
			loc.append(l(x).permute(0, 2, 3, 1).contiguous())
			conf.append(c(x).permute(0, 2, 3, 1).contiguous())

		loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
		conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

		loc = loc.view(loc.size(0), -1, 4)
		conf = conf.view(conf.size(0), -1, self.num_classes)

		output = (loc, conf, self.dbox_list)

		if self.phase == "inference":
			return self.detect(output[0], output[1], output[2])
		else:
			return output

def decode(loc, defbox_list):
	boxes = torch.cat((defbox_list[:, :2] + 0.1*loc[:, :2]*defbox_list[:, 2:],
		defbox_list[:, 2:]*torch.exp(loc[:, 2:]*0.2)), dim=1)
	boxes[:, :2] -= boxes[:, 2:]/2
	boxes[:, 2:] += boxes[:, :2]

	return boxes

def nms(boxes, scores, overlap=0.45, top_k=200):
	count = 0
	keep = scores.new(scores.size()).zero_().long()

	# boxes coordinate
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# area of boxes
	area = torch.mul(x2-x1, y2-y1)

	tmp_x1 = boxes.new()
	tmp_y1 = boxes.new()
	tmp_x2 = boxes.new()
	tmp_y2 = boxes.new()
	tmp_w = boxes.new()
	tmp_h = boxes.new()

	value, idx = scores.sort(0)
	idx = idx[-top_k:] # id of top 200 boxes with confidence max

	while idx.numel() > 0:
		i = idx[-1] # id of bording box with confidence max
		keep[count] = i
		count += 1

		if idx.size(0) == 1:
			break
		idx = idx[:-1] # id of boxes tru box co do tu tin max

		# information box
		torch.index_select(x1, 0, idx, out=tmp_x1)
		torch.index_select(y1, 0, idx, out=tmp_y1)
		torch.index_select(x2, 0, idx, out=tmp_x2)
		torch.index_select(y2, 0, idx, out=tmp_y2)

		tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
		tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
		tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
		tmp_y2 = torch.clamp(tmp_y2, min=y2[i])

		# chuyen ve tensor co size ma index giam di 1
		tmp_w.resize_as_(tmp_x2)
		tmp_h.resize_as_(tmp_y2)
		tmp_w = tmp_x2-tmp_x1
		tmp_h = tmp_y2-tmp_y1

		tmp_w = torch.clamp(tmp_w, min=0.0)
		tmp_h = torch.clamp(tmp_h, min=0.0)

		inter = tmp_w*tmp_h
		others_area = torch.index_select(area, 0, idx)
		union = area[i] + others_area - inter

		iou = inter/union

		idx = idx[iou.le(overlap)]

	return keep, count

class Detect(Function):
	def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
		self.softmax = nn.Softmax(dim=-1)
		self.conf_thresh = conf_thresh
		self.top_k = top_k
		self.nms_thresh = nms.nms_thresh

	def forward(self, loc_data, conf_data, dbox_list):
		num_batch = loc_data.size(0) # batch size
		num_dbox = loc_data.size(1) # 8732
		num_classes = conf_data.size(2) # 21

		conf_data = self.softmax(conf_data)
		conf_preds = conf_data.transpose(2, 1)

		output = torch.zeros(num_batch, num_classes, self.top_k, 5)

		# xu ly tung buc anh trong 1 bach cac buc anh
		for i in range(num_batch):
			# tinh bbox tu offset information va default box
			decode_boxes = decode(loc_data[i], dbox_list)

			# copy confidence score cua anh thu i
			conf_scores = conf_preds[i].clone()

			for cl in range(1, num_classes):
				c_mask = conf_preds[cl].gt(self.conf_thresh) # chi lay nhung confidence > 0.01
				scores = conf_preds[cl][c_mask]

				if scores.element() == 0:
					continue

				# dua chieu ve giong chieu cua decode_boxes de tinh toan
				l_mask = c_mask.unsquzee(1).enpand_as(decode_boxes)

				boxes = decode_boxes[l_mask].view(-1, 4)

				ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

				output[i, cl, :count] = torch.cat((scores[ids[:count]].unsquzee(1), boxes[ids[:count]]), 1)

		return output

# if __name__ == '__main__':
# 	# vgg = create_vgg()
# 	# print(vgg)
# 	# extras = create_extras()
# 	# print(extras)

# 	# loc, conf = create_loc_conf()
# 	# print(loc)
# 	# print(conf)
# 	ssd = SSD(phase="train", cfg=cfg)
# 	print(ssd)
