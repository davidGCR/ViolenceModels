import torch
import torch.nn as nn
import numpy as np

def tempMaxPooling(stacked_images):
    num_dynamic_images = stacked_images.size()[0]
    tmppool = nn.MaxPool2d((num_dynamic_images, 1))
    spermute = stacked_images.permute(2,3,0,1)
    out = tmppool(spermute)
    out = out.permute(2,3,0,1)
    return out

def main():
    t1=torch.tensor([[[1,2,3],[4,5,6],[7,6,9]],[[10,20,30],[40,50,60],[70,60,90]]])
    t1 = t1*2.3
    t1 = t1.float()
    
    t2=t1*2
    t2 = t2*0.3
    t2 = t2.float()
    
    t3=t1*3
    t3 = t3*0.7
    t3 = t3.float()
    
    t4=t1*4
    t4 = t4*0.005
    t4 = t4.float()
    lista = [t1,t2,t3,t4]
    stack = torch.stack(lista,0)
    
    torch.set_printoptions(precision=10,sci_mode=False)

    print('stack size: ',stack.size())
    print(stack)
    out = tempMaxPooling(stack)
    print('out size: ',out.size())
    print(out)

if __name__ == "__main__":
    main()
    