import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
        

        return img, labels, cont_labels, name

class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= 1000 and abs((label[1]*180/np.pi)) <= 1000:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 1000 and abs((label[1]*180/np.pi)) <= 1000:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-180, 180,4))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name


class Mpiigaze2(Dataset): 
  def __init__(self, pathorg, root, transform, train, fold, path2, root2):
    self.root2 = root2
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= 90 and abs((label[1]*180/np.pi)) <= 90:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 90 and abs((label[1]*180/np.pi)) <= 90:
                self.lines.append(line)
    self.len = len(self.lines)
    if isinstance(path2, list):
        for i in path2:
            with open(i) as f:
                line = f.readlines()
                line.pop(0)
                self.lines.extend(line)
    else:
        with open(path2) as f:
            lines = f.readlines()
            lines.pop(0)
            self.orig_list_len = len(lines)
            count = 0
            max_lines = 42000
            count2 = 0
            for line in lines:
                gaze2d = line.strip().split(" ")[5]
                label = np.array(gaze2d.split(",")).astype("float")
                if abs(label[0] * 180 / np.pi) <= 60 and abs(label[1] * 180 / np.pi) <= 60:
                    self.lines.append(line)
                    count += 1
                    if count >= max_lines:
                        break
                else:
                   count2 += 1
   
    print("{} items removed from dataset that have an angle > {}".format(count2,60))
        
  def __len__(self):
    return self.len + 25000

  def __getitem__(self, idx):
    if idx < self.len:
      line = self.lines[idx]
      line = line.strip().split(" ")

      name = line[3]
      gaze2d = line[7]
      face = line[0]

      label = np.array(gaze2d.split(",")).astype("float")
      label = torch.from_numpy(label).type(torch.FloatTensor)


      pitch = label[0]* 180 / np.pi
      yaw = label[1]* 180 / np.pi

      img = Image.open(os.path.join(self.root, face))

      # fimg = cv2.imread(os.path.join(self.root, face))
      # fimg = cv2.resize(fimg, (448, 448))/255.0
      # fimg = fimg.transpose(2, 0, 1)
      # img=torch.from_numpy(fimg).type(torch.FloatTensor)
      
      if self.transform:
          img = self.transform(img)        
      
      # Bin values
      bins = np.array(range(-180, 180,4))
      binned_pose = np.digitize([pitch, yaw], bins) - 1

      labels = binned_pose
      cont_labels = torch.FloatTensor([pitch, yaw])
    else:
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]

        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)
        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi
        img = Image.open(os.path.join(self.root2, face))
        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)
        if self.transform:
            img = self.transform(img)        
        # Bin values
        bins = np.array(range(-180, 180, 4))
        binned_pose = np.digitize([pitch, yaw], bins) - 1
        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
    return img, labels, cont_labels, name
  
class Mpiigaze3(Dataset): 
  def __init__(self, pathorg, root, transform, transform2, train, angle,fold=0):
    self.transform = transform
    self.transform2 = transform2
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= 1000 and abs((label[1]*180/np.pi)) <= 1000:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 1000 and abs((label[1]*180/np.pi)) <= 1000:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d_left = line[7]
    gaze2d_right = line[8]
    face = line[0]
    left = line[1]
    right = line[2]
    label_left = np.array(gaze2d_left.split(",")).astype("float")
    label_left = torch.from_numpy(label_left).type(torch.FloatTensor)
    label_right = np.array(gaze2d_right.split(",")).astype("float")
    label_right = torch.from_numpy(label_right).type(torch.FloatTensor)


    pitch_left = label_left[0]* 180 / np.pi
    yaw_left = label_left[1]* 180 / np.pi
    pitch_right = label_right[0]* 180 / np.pi
    yaw_right = label_right[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))
    left_img = Image.open(os.path.join(self.root, left))
    right_img = Image.open(os.path.join(self.root, right))
    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)     
        left_img = self.transform2(left_img)
        right_img = self.transform2(right_img)   
    
    # Bin values
    bins = np.array(range(-180, 180,4))
    labels_left = np.digitize([pitch_left, yaw_left], bins) - 1
    labels_right = np.digitize([pitch_right, yaw_right], bins) - 1
    cont_labels_left = torch.FloatTensor([pitch_left, yaw_left])
    cont_labels_right = torch.FloatTensor([pitch_right, yaw_right])

    return img, left_img, right_img, labels_left, cont_labels_left, labels_right, cont_labels_right, name