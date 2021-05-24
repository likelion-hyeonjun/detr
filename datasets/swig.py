from __future__ import print_function, division

from typing import List

import torch
import numpy as np
import random
import csv
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data.sampler import Sampler

import json
from PIL import Image
import util.misc as utils
torch.multiprocessing.set_sharing_strategy('file_system')


class imSituDataset(Dataset):

    def __init__(self, img_folder, train_file, noun_list, verb_path, role_path, verb_info, inference=False, inference_verbs=None, transform=None):

        self.img_folder = img_folder
        self.inference = inference
        self.inference_verbs = inference_verbs
        self.train_file = train_file
        self.verb_path = verb_path
        self.role_path = role_path
        self.noun_path = noun_list
        self.transform = transform

        with open(self.verb_path, 'r') as f:
            self.verb_to_idx, self.idx_to_verb = self.load_verb(f)
            self.num_verbs = len(self.verb_to_idx)
        with open(self.role_path, 'r') as f:
            self.role_to_idx, self.idx_to_role = self.load_role(f)
            self.num_roles = len(self.role_to_idx)
        with open(self.noun_path, 'r') as file:
            self.noun_to_idx, self.idx_to_noun = self.load_nouns(csv.reader(file, delimiter=','))
            self.num_nouns = len(self.noun_to_idx) - 1  # padding noun last
            self.pad_noun = len(self.noun_to_idx) - 1

        # verb_role
        self.verb_role = {verb: value['order'] for verb, value in verb_info.items()}
        self.vidx_ridx = [[self.role_to_idx[role] for role in self.verb_role[verb]] for verb in self.idx_to_verb]

        if not self.inference:
            with open(self.train_file) as file:
                train_json = json.load(file)
            self.image_data = self._read_annotations(train_json, self.noun_to_idx)
            self.image_names = list(self.image_data.keys())
        else:
            self.image_names = []
            with open(train_file) as f:
                for line in f:
                    self.image_names.append(line.split('\n')[0])

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

        # verb role adjacency matrix
        self.verb_role_adj_matrix = np.tile(np.identity(len(self.role_to_idx)),
                                            (len(self.verb_to_idx), 1, 1)).astype(bool)
        for vidx, ridx in enumerate(self.vidx_ridx):
            ridx = np.array(ridx)
            self.verb_role_adj_matrix[vidx:vidx + 1, ridx[:, None], ridx] = np.ones(len(ridx)).astype(bool)

        # role adjacency matrix
        self.role_adj_matrix = self.verb_role_adj_matrix.any(0)

    def load_nouns(self, csv_reader):
        result = {}
        idx_to_result = []
        for line, row in enumerate(csv_reader):
            line += 1
            noun_name, noun_id = row
            noun_id = int(noun_id)
            if noun_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, noun_name))
            result[noun_name] = noun_id
            idx_to_result.append(noun_name.split('_')[0])

        return result, idx_to_result

    def load_verb(self, file):
        verb_to_idx = {}
        idx_to_verb = []

        k = 0
        for line in file:
            verb = line.split('\n')[0]
            idx_to_verb.append(verb)
            verb_to_idx[verb] = k
            k += 1
        return verb_to_idx, idx_to_verb

    def load_role(self, file):
        role_to_idx = {}
        idx_to_role = []

        k = 0
        for line in file:
            role = line.split('\n')[0]
            idx_to_role.append(role)
            role_to_idx[role] = k
            k += 1
        return role_to_idx, idx_to_role

    def make_dummy_annot(self):
        return np.zeros((len(self.role_to_idx), 3))

    def __len__(self):
        # return 16
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        if self.inference:
            verb_idx = self.inference_verbs[self.image_names[idx]]
            annot = self.make_dummy_annot()
            sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
            if self.transform:
                sample['img'] = self.transform(sample['img'])
            return sample

        annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[2]
        verb = verb.split('_')[0]

        verb_idx = self.verb_to_idx[verb]
        verb_role_idx = self.vidx_ridx[verb_idx]
        sample = {'img': img, 'annot': annot,
                  'img_name': self.image_names[idx], 'verb_idx': verb_idx, 'verb_role_idx': verb_role_idx}
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        return sample

    def load_image(self, image_index):

        im = Image.open(self.image_names[image_index])
        im = im.convert('RGB')
        return im

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 3))

        # parse annotations
        for a in annotation_list:
            annotation = np.zeros((1, 3))  # allow for 3 annotations

            annotation[0, 0] = self.noun_to_idx[a['class1']]
            annotation[0, 1] = self.noun_to_idx[a['class2']]
            annotation[0, 2] = self.noun_to_idx[a['class3']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, json, classes):
        dummy_annot = [{f"class{n}": 'Pad' for n in [1, 2, 3]} for role in self.role_to_idx.keys()]

        result = {}
        for image in json:
            verb = json[image]['verb']
            frames = json[image]['frames']
            img_file = f"{self.img_folder}/" + image

            result[img_file] = dummy_annot.copy()
            for role in self.verb_role[verb]:
                class1 = frames[0][role]
                class2 = frames[1][role]
                class3 = frames[2][role]
                if class1 == '':
                    class1 = 'blank'
                if class2 == '':
                    class2 = 'blank'
                if class3 == '':
                    class3 = 'blank'
                if class1 not in classes:
                    class1 = 'oov'
                if class2 not in classes:
                    class2 = 'oov'
                if class3 not in classes:
                    class3 = 'oov'

                ridx = self.role_to_idx[role]
                result[img_file][ridx] = {
                    'class1': class1,
                    'class2': class2,
                    'class3': class3
                }
        return result

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    return (utils.nested_tensor_from_tensor_list([s['img'] for s in data]),
            [{'verb': torch.tensor(s['verb_idx']),
              'roles': torch.tensor(s['verb_role_idx']),
              'img_name': s['img_name'],
              'labels': torch.tensor(s['annot'])}
             for s in data])


class MinMaxResizeScale(torch.nn.Module):
    def __init__(self, min_edge: int, max_edge: int, scales: List[int] = [1.0, 0.75, 0.5]):
        super().__init__()
        assert min_edge < max_edge
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.scales = scales

    def forward(self, img):
        min_edge = min(img.size)
        max_edge = max(img.size)
        scale = random.choice(self.scales)

        # (min, max) -> (min_edge, max * min_edge / max)
        # resized_max = (max_edge * self.min_edge) / min_edge
        if (max_edge * self.min_edge) > (self.max_edge * min_edge):
            # resized_max > max_edge
            size = int(round((min_edge * self.max_edge * scale) / max_edge))
            if img.size[0] < img.size[1]:
                # F.resize(w x h, [h, w])
                return F.resize(img, [int(round(self.max_edge*scale)), size])
            else:
                return F.resize(img, [size, int(round(self.max_edge*scale))])
        else:
            return F.resize(img, self.min_edge)


class MinMaxResize(torch.nn.Module):
    def __init__(self, min_edge: int, max_edge: int):
        super().__init__()
        assert min_edge < max_edge
        self.min_edge = min_edge
        self.max_edge = max_edge

    def forward(self, img):
        min_edge = min(img.size)
        max_edge = max(img.size)
        # (min, max) -> (min_edge, max * min_edge / max)
        # resized_max = (max_edge * self.min_edge) / min_edge
        if (max_edge * self.min_edge) > (self.max_edge * min_edge):
            # resized_max > max_edge
            size = round((min_edge * self.max_edge) / max_edge)
            if img.size[0] < img.size[1]:
                # F.resize(w x h, [h, w])
                return F.resize(img, [self.max_edge, size])
            else:
                return F.resize(img, [size, self.max_edge])
        else:
            return F.resize(img, self.min_edge)


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


def build(image_set, args):
    root = Path(args.swig_path)
    img_folder = root / args.image_dir

    PATHS = {
        "train": root / "SWiG_jsons" / "train.json",
        "val": root / "SWiG_jsons" / "dev.json",
        "test": root / "SWiG_jsons" / "test.json",
    }
    ann_file = PATHS[image_set]

    classes_file = Path(args.swig_path) / "SWiG_jsons" / "train_classes.csv"
    verb_path = Path(args.swig_path) / "SWiG_jsons" / "verb_indices.txt"
    role_path = Path(args.swig_path) / "SWiG_jsons" / "role_indices.txt"

    with open(f'{args.swig_path}/SWiG_jsons/imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']

    color = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomGrayscale(p=0.3)])
    rotation = transforms.RandomRotation(0)
    hflip = transforms.RandomHorizontalFlip()

    resize = MinMaxResize(256, 350)
    resize_scale = MinMaxResizeScale(256, 350, [1.0, 0.75, 0.5])
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    TRANSFORMS = {
        "train": transforms.Compose([color, rotation, hflip,
                                     resize_scale, to_tensor, normalizer]),
        "val": transforms.Compose([resize, to_tensor, normalizer]),
        "test": transforms.Compose([resize, to_tensor, normalizer]),
    }
    tfs = TRANSFORMS[image_set]

    dataset = imSituDataset(img_folder=str(img_folder),
                            train_file=ann_file,
                            noun_list=classes_file,
                            verb_path=verb_path,
                            role_path=role_path,
                            verb_info=verb_orders,
                            transform=tfs)
    args.vr_adj_mat = dataset.verb_role_adj_matrix
    args.r_adj_mat = dataset.role_adj_matrix
    args.num_verbs = dataset.num_verbs
    args.num_roles = dataset.num_roles
    args.num_nouns = dataset.num_nouns
    args.pad_noun = dataset.pad_noun

    return dataset
