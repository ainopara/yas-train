import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import ImageFont
import os

from mona.datagen.datagen import DataGen
from mona.config import config
from mona.nn import predict as predict_net
from mona.nn.model2 import Model2
from mona.text import get_lexicon

import datetime

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

lexicon = get_lexicon(config["model_type"])

# 4k分辨率最大对应84号字，900p分辨率最小对应18号字
if config["model_type"] == "Genshin":
    fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "StarRail":
    fonts = [ImageFont.truetype("./assets/starrail.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "WutheringWaves":
    fonts = [ImageFont.truetype("./assets/wuthering_waves/ARFangXinShuH7GBK-HV.ttf", i) for i in range(15, 90)]
datagen = DataGen(config, fonts, lexicon)

print("lexicon size: ", lexicon.lexicon_size())

# a list of target strings
def get_target(s):
    target_length = []

    target_size = 0
    for i, target in enumerate(s):
        target_length.append(len(target))
        target_size += len(target)

    target_vector = []
    for target in s:
        for char in target:
            index = lexicon.word_to_index[char]
            if index == 0:
                print("error")
            target_vector.append(index)

    target_vector = torch.LongTensor(target_vector)
    target_length = torch.LongTensor(target_length)

    return target_vector, target_length


def validate(net, validate_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            predict = predict_net(net, x, lexicon)
            # print(predict)
            correct += sum([1 if predict[i] == label[i] else 0 for i in range(len(label))])
            total += len(label)

    net.train()
    return correct / total


def validate_custom(net):
    """Validate the model on custom validation set from extra training data."""
    if not os.path.exists("data/custom_validate_x.pt") or not os.path.exists("data/custom_validate_label.pt"):
        print("No custom validation data found")
        return None
    
    try:
        custom_x = torch.load("data/custom_validate_x.pt")
        custom_y = torch.load("data/custom_validate_label.pt")
        
        if len(custom_y) == 0:
            print("Custom validation set is empty")
            return None
        
        custom_dataset = MyDataSet(custom_x, custom_y)
        custom_loader = DataLoader(custom_dataset, batch_size=config["batch_size"])
        
        net.eval()
        correct = 0
        total = 0
        failed_cases = []
        
        with torch.no_grad():
            for x, label in custom_loader:
                x = x.to(device)
                predict = predict_net(net, x, lexicon)
                
                for i in range(len(label)):
                    if predict[i] == label[i]:
                        correct += 1
                    else:
                        failed_cases.append((predict[i], label[i]))
                total += len(label)
        
        net.train()
        accuracy = correct / total if total > 0 else 0
        
        print(f"Custom validation: {correct}/{total} correct ({accuracy * 100:.2f}%)")
        if failed_cases and len(failed_cases) <= 5:  # Show up to 5 failed cases
            print("Failed cases:")
            for pred, truth in failed_cases:
                print(f"  Predicted: '{pred}' | Truth: '{truth}'")
        
        return accuracy
        
    except Exception as e:
        print(f"Error loading custom validation data: {e}")
        return None


def train():
    # net = Model(len(index_to_word)).to(device)
    # net = Model2(len(index_to_word), 1, hidden_channels=128, num_heads=4).to(device)

    # 这是现在在用的model
    # model2就是SVTR（非原版）
    net = Model2(lexicon.lexicon_size(), 1).to(device)

    # net = SVTRNet(
    #     img_size=(32, 384),
    #     in_channels=1,
    #     out_channels=len(index_to_word)
    # ).to(device)
    if config["pretrain"]:
        # assume the old index_to_word is in "models/index_2_word.json"
        net.load_state_dict(torch.load(f"models/{config['pretrain_name']}"))
        # net.load_can_load(torch.load(f"models/{config['pretrain_name']}"))

    data_aug_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(1, 1),
                transforms.GaussianBlur(3, 3),
                transforms.GaussianBlur(5, 5),
                # transforms.GaussianBlur(7,7),
            ])], p=0.5),

        transforms.RandomApply([
            transforms.RandomCrop(size=(31, 383)),
            transforms.Resize((32, 384), antialias=True),
            ], p=0.5),

        transforms.RandomApply([AddGaussianNoise(mean=0, std=1/255)], p=0.5),
    ])

    train_dataset = MyOnlineDataSet(config['train_size']) if config["online_train"] else MyDataSet(
        torch.load("data/train_x.pt"), torch.load("data/train_label.pt"))
    validate_dataset = MyOnlineDataSet(config['validate_size']) if config["online_val"] else MyDataSet(
        torch.load("data/validate_x.pt"), torch.load("data/validate_label.pt"))

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=config["dataloader_workers"], batch_size=config["batch_size"],)
    validate_loader = DataLoader(validate_dataset, num_workers=config["dataloader_workers"], batch_size=config["batch_size"])

    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer = optim.Adadelta(net.parameters())
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)

    epoch = config["epoch"]
    print_per = config["print_per"]
    save_per = config["save_per"]
    batch = 1
    # net.freeze_backbone()
    start_time = datetime.datetime.now()
    for epoch in range(epoch):
        if epoch == config["unfreeze_backbone_epoch"]:
            pass
            # net.unfreeze_backbone()
        for x, label in train_loader:
            optimizer.zero_grad()
            target_vector, target_lengths = get_target(label)
            target_vector, target_lengths = target_vector.to(device), target_lengths.to(device)
            x = x.to(device)

            # Data Augmentation in batch
            x = data_aug_transform(x)

            batch_size = x.size(0)

            y = net(x)

            input_lengths = torch.full((batch_size,), 24, device=device, dtype=torch.long)
            loss = ctc_loss(y, target_vector, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            cur_time = datetime.datetime.now()

            if batch % print_per == 0:
                tput = batch_size * batch / (cur_time - start_time).total_seconds()
                print(f"{cur_time} e{epoch} #{batch} tput: {tput} loss: {loss.item()}")

            if batch % save_per == 0:
                print("Validating and checkpointing")
                rate = validate(net, validate_loader)
                print(f"{cur_time} rate: {rate * 100}%")
                
                # Custom validation on extra training data
                custom_rate = validate_custom(net)
                if custom_rate is not None:
                    print(f"{cur_time} custom rate: {custom_rate * 100:.2f}%")
                
                torch.save(net.state_dict(), f"models/model_training.pt")
                if rate == 1:
                    torch.save(net.state_dict(), f"models/model_acc100-epoch{epoch}.pt")

            batch += 1

    for x, label in validate_loader:
        x = x.to(device)
        # predict = net.predict(x)
        predict = predict_net(net, x, lexicon)
        print("predict:     ", predict[:10])
        print("ground truth:", label[:10])
        break
    
    # Final custom validation
    print("\nFinal custom validation:")
    final_custom_rate = validate_custom(net)
    if final_custom_rate is not None:
        print(f"Final custom validation accuracy: {final_custom_rate * 100:.2f}%")


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MyDataSet(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.x[index]
        label = self.labels[index]
        return x, label

class MyOnlineDataSet(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Generate data online
        im, text = datagen.generate_image()
        tensor = transforms.ToTensor()(im)
        return tensor, text
