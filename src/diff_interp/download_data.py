import torchvision 

if __name__ == "__main__":
    dataset = torchvision.datasets.CelebA(root='./data', download=True)