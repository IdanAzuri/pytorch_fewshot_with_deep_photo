import torch
import torchvision.transforms as transforms
from torchvision import datasets
from hbconfig import Config


def get_loader(mode):
	"""Builds and returns Dataloader for MNIST and SVHN dataset."""
	config = Config
	transform_list = []
	is_train = mode == "train"
	
	if config.model.use_augmentation:
		transform_list.append(transforms.RandomHorizontalFlip())
		transform_list.append(transforms.RandomRotation(0.1))
	
	loader= None
	transform = transforms.Compose([transforms.Resize(config.data.image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if config.model.dataset == "mnist":
		mnist = datasets.MNIST(root=config.data.mnist_path, download=True, transform=transform, train=is_train)
		loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "svhn":
		svhn = datasets.SVHN(root=config.data.svhn_path, download=True, transform=transform, split=mode)
		loader = torch.utils.data.DataLoader(dataset=svhn, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	
	
	
	
	
	
	## preparing for AC costum dataset
	# train_size = int(0.8 * len(full_dataset))
	# test_size = len(full_dataset) - train_size
	# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
	return loader
