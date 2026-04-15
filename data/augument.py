from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),   
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])