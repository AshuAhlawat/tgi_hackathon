def predict(img_path):
    img_cls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    import cv2
    import torch
    import torch.nn as nn
    from PIL import Image
    import numpy as np

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                # 3 128
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                # 32 64
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, 2),
                # 64 32
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                # 128 16
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, 2),
                # 128 8
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                # 256 4
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, 2),
                # 256 2

                nn.Flatten(),
                # 256*2*2
                nn.Linear(256 * 2 * 2, 128),
                nn.ReLU(),
                # 128
                nn.Linear(128, 26)
                # 26
            )

        def forward(self, inputs):

            out = self.network(inputs)
            return torch.softmax(out, dim=-1)

    model = Model()

    path = "./eyensign/static/sign_model97.pth"
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path))
            print("Model Loaded(GPU)")
        else:
            model.load_state_dict(torch.load(path,map_location=torch.device("cpu")))
            print("Model Loaded(CPU)")
    except Exception as e:
        pass

    img = Image.open(img_path)
    img = img.resize((128,128))
    img.show()

    img_arr = np.asarray(img)
    
    img_arr = img_arr/255
    img_tsr = torch.Tensor([img_arr])
    img_tsr = img_tsr.permute(0,3,1,2)
    
    pred = model(img_tsr).detach()
    pred = np.array(pred[0])
    pred_index = np.where(pred==max(pred))[0][0]
    
    print("\n\nPrediction: ",img_cls[pred_index])

    # return img_cls[pred_index]


predict("./val/F/10.jpg")