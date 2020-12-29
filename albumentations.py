import albumentations as A

#基本的な使い方

#BGR読み込み
image_bgr = cv2.imread\
            ('../input/cassava-leaf-disease-classification/train_images/1000723321.jpg', 1) 
img_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)



#albumentationsで出力を調整する場合。
# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

#出力確認
val = transform(image=img_rgb)
#定義したtransformだけではなく、["image"]を適用することが必要
img1=val["image"]
print(img1.shape)
plt.imshow(img1)
plt.show()


#kaggleで参考にしたもの
import albumentations

out_size=224

# augmentations taken from: https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
train_aug = albumentations.Compose([
            albumentations.RandomResizedCrop(out_size,out_size),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)], p=1.)
  
        
valid_aug = albumentations.Compose([
            albumentations.CenterCrop(256, 256, p=1.),
            albumentations.Resize(out_size,out_size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )], p=1.)
            

class My_Dataset(Dataset):
    def __init__(self,df,transform):
        #dataframeを格納
        self.df = df
        self.img_path =pathlib.Path("../input/cassava-leaf-disease-classification/train_images")
        
        #transform
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        #image_id
        img_id=self.df.iloc[index,0]
        
        #open cv2
        img_bgr=cv2.imread(str(self.img_path/img_id))  #BGR
        img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        #transform with albumentations
        if self.transform is not None:
            img_transformed=self.transform(image=img_rgb)["image"]
        
        #change channels first
        img=np.einsum('ijk->kij', img_transformed)
        
        #label
        target=self.df.iloc[index,1]
        #target
        
        return img,target
        
#DataSet
train_dataset=My_Dataset(df_train,transform=train_aug)
val_dataset=My_Dataset(df_val,transform=valid_aug)

#Dataloader
batch_size=10
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

#dict
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#test sample
batch_iterator=iter(dataloaders_dict["train"])
inputs,labels=next(batch_iterator)
print(inputs.shape) 
print(labels.shape)
print(labels)
#出力可能
