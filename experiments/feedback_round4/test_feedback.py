import unittest
import numpy as np
import torch
from core import masked_loss, training_pair, predict, s

class ConstantVelocity(torch.nn.Module):
    def forward(self,x,t,c): return torch.ones_like(x)*0.25

class FeedbackTests(unittest.TestCase):
    def test_masked_objectives_ignore_trace(self):
        a=torch.zeros(1,2,2,2,requires_grad=True); b=torch.tensor([[[[2.,99.],[0.,0.]],[[2.,99.],[0.,0.]]]])
        mask=torch.tensor([[[False,True],[False,False]]])
        self.assertAlmostEqual(masked_loss(a,b,mask,'l2').item(),8/6,places=6)
        self.assertAlmostEqual(masked_loss(a,b,mask,'l1').item(),4/6,places=6)
        masked_loss(a,b,mask,'l1').backward(); self.assertTrue((a.grad[:,:,0,1]==0).all())
    def test_flow_training_path_endpoints(self):
        torch.manual_seed(1); target=torch.ones(2,2,64,128)*.2; mask=torch.zeros(2,64,128,dtype=torch.bool); mask[:,4,5]=True
        x,t,v=training_pair(target*(~mask)[:,None],mask,'flow',None)
        end=x+(1-t[:,None,None,None]/999)*v
        torch.testing.assert_close(end,target*(~mask)[:,None],atol=5e-7,rtol=1e-5)
    def test_euler_constant_velocity_and_target_independence(self):
        b={'x':torch.zeros(2,4),'mask':torch.zeros(2,64,128,dtype=torch.bool),'u':torch.randn(2,2,64,128)}
        b['mask'][:,3,4]=True; model=ConstantVelocity(); mean=torch.zeros(1,2,1,1); std=torch.ones_like(mean)*.01
        p=predict(model,b,np.array([5,9]),6180,mean,std,'flow',20)
        expected=((s.initial_noise([5,9],6180,'cpu')+.25)*.01).masked_fill(b['mask'][:,None],0)
        torch.testing.assert_close(p,expected,atol=1e-7,rtol=1e-5)
        b['u']*=999
        torch.testing.assert_close(p,predict(model,b,np.array([5,9]),6180,mean,std,'flow',20),rtol=0,atol=0)
    def test_ddim_independent_of_ground_truth(self):
        b={'x':torch.zeros(2,4),'mask':torch.zeros(2,64,128,dtype=torch.bool),'u':torch.randn(2,2,64,128)}
        model=ConstantVelocity(); mean=torch.zeros(1,2,1,1);std=torch.ones_like(mean)
        a=predict(model,b,np.array([5,9]),6180,mean,std,'diffusion',20); b['u']*=123
        torch.testing.assert_close(a,predict(model,b,np.array([5,9]),6180,mean,std,'diffusion',20),rtol=0,atol=0)
    def test_noise_row_order_and_batch_invariance(self):
        a=s.initial_noise([3,7,11],6180,'cpu'); b=torch.cat([s.initial_noise([11],6180,'cpu'),s.initial_noise([3,7],6180,'cpu')])
        torch.testing.assert_close(a,b[[1,2,0]],rtol=0,atol=0)
    def test_field_metrics_sanity(self):
        truth=np.ones((2,64,128)); mask=np.zeros((64,128),bool);mask[31,60:68]=True
        y=np.linspace(-1,1,128);z=np.linspace(0,8,64)
        self.assertEqual(s.metrics.design_metrics(truth,truth,mask,y,z)['relative_l2'],0)
        self.assertAlmostEqual(s.metrics.design_metrics(np.zeros_like(truth),truth,mask,y,z)['relative_l2'],1)

if __name__=='__main__':unittest.main()
