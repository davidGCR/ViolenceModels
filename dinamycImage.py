import numpy as np
import Image from PIL

def getDynamicImage(frames):
    frames = np.stack(frames, axis=0)
    fw = np.zeros(self.seqLen)  
    for i in range(self.seqLen): #frame by frame
      fw[i] = np.sum( np.divide((2*np.arange(i+1,self.seqLen+1)-self.seqLen-1) , np.arange(i+1,self.seqLen+1))  )

    fwr = fw.reshape(self.seqLen,1,1,1)
    sm = frames*fwr
    sm = sm.sum(0)
    sm = sm - np.min(sm) 
    sm = 255 * sm /np.max(sm) 
    img = sm.astype(np.uint8)
    ##to PIL image
    img = Image.fromarray(np.uint8(img))
    return img