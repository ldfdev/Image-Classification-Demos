import PyTorch_tl_inception_v3_featureExtract
import torch, torchvision
import WhaleDataset

if __name__=='__main__':
    model = torchvision.models.vgg16(pretrained=True)
    # pretrained is used only for initialization, no weights freezing
    # for item in vars(model).items():
    #     print(item)
    # for param in model.parameters():
    #     param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=8)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    model = model.to(device)
    optimizer_v3 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # model.classifier[-1].parameters()
    # dataset = WhaleDataset.PatchesWhaleDataset()
    dataset = PyTorch_tl_inception_v3_featureExtract.WhaleDataset()
    model = PyTorch_tl_inception_v3_featureExtract.train_model()