"""Check the largest requested batch on the allocated GPU before the screen."""
import json, torch
from core import s,masked_loss
s.setup(6180,'cuda'); model=s.make_model('paired',36).cuda()
x=torch.randn(32,2,64,128,device='cuda');t=torch.randint(1000,(32,),device='cuda');c=torch.rand(32,4,device='cuda')
mask=torch.zeros(32,64,128,device='cuda',dtype=torch.bool)
opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
loss=masked_loss(model(x,t,c),torch.randn_like(x),mask)
loss.backward();opt.step();torch.cuda.synchronize()
print(json.dumps({'batch_size':32,'base':36,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'loss':loss.item(),'status':'passed'}))
