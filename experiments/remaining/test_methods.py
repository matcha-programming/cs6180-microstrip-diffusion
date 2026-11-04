"""Independent checks for masking, conditioning, and DDIM algebra."""
import unittest
import numpy as np
import torch
from fourier import FourierDeepONet
from models import ConditionalUNet,schedule,masked_noise_loss,ddim_update
from diffusion import train_statistics,sample_batch


class Methods(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(6180)

    def test_fixed_fourier_matrix(self):
        a=FourierDeepONet(scale=.5);b=FourierDeepONet(scale=1)
        self.assertEqual(tuple(a.B.shape),(128,2));self.assertTrue(torch.equal(a.B*2,b.B))
        self.assertNotIn('B',dict(a.named_parameters()))
        self.assertEqual(tuple(a(torch.randn(2,4),torch.randn(2,17,2)).shape),(2,2,17))

    def test_noise_loss_excludes_mask(self):
        mask=torch.zeros(2,8,8,dtype=torch.bool);mask[:,3:5,3:5]=True
        noise=torch.randn(2,2,8,8);pred=noise.clone();pred.masked_fill_(mask[:,None],1e6)
        self.assertEqual(float(masked_noise_loss(pred,noise,mask)),0)

    def test_ddim_oracle(self):
        a=schedule('cpu');clean=torch.randn(2,2,8,8);eps=torch.randn_like(clean)
        mask=torch.zeros(2,8,8,dtype=torch.bool);mask[:,3,4]=True
        noisy=a[500].sqrt()*clean+(1-a[500]).sqrt()*eps
        expected=a[100].sqrt()*clean+(1-a[100]).sqrt()*eps
        actual=ddim_update(noisy,eps,a[500],a[100],mask)
        torch.testing.assert_close(actual,expected.masked_fill(mask[:,None],0),atol=2e-6,rtol=2e-6)
        final=ddim_update(noisy,eps,a[500],a.new_tensor(1.),mask)
        torch.testing.assert_close(final,clean.masked_fill(mask[:,None],0),atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.all(a[1:]<a[:-1]))

    def test_training_statistics_ignore_holdout_and_mask(self):
        data=torch.randn(4,2,8,8);mask=torch.zeros(4,8,8,dtype=torch.bool);mask[:,3,4]=True
        before=train_statistics(data,mask,np.array([0,1]));data[2:]=1e8;data.masked_fill_(mask[:,None],-1e8)
        after=train_statistics(data,mask,np.array([0,1]))
        for x,y in zip(before,after):torch.testing.assert_close(x,y)

    def test_unet_conditioning_and_shape(self):
        model=ConditionalUNet(base=4);torch.nn.init.normal_(model.output[-1].weight,std=.1)
        x=torch.randn(2,2,64,128);t=torch.tensor([500,500]);design=torch.randn(2,4)
        pred=model(x,t,design);self.assertEqual(pred.shape,x.shape)
        self.assertFalse(torch.allclose(pred,model(x,t,design+1)))
        pred.square().mean().backward();self.assertIsNotNone(model.design_mlp[0].weight.grad)
        residual=ConditionalUNet(residual=True,base=4)
        self.assertEqual(residual(x,t,design,torch.zeros_like(x)).shape,x.shape)

    def test_sampler_clamps_mask(self):
        model=ConditionalUNet(base=4).eval();mask=torch.zeros(2,64,128,dtype=torch.bool);mask[:,31:34,50:78]=True
        data={'mask':mask,'x':torch.randn(2,4)};ids=torch.tensor([0,1]);mean=torch.zeros(1,2,1,1);std=torch.ones_like(mean)
        g=torch.Generator().manual_seed(6180)
        out=sample_batch(model,data,ids,None,None,mean,std,schedule('cpu'),3,g)
        self.assertTrue(torch.all(out.masked_select(mask[:,None])==0));self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(float(out.abs().max()),1)

    def test_full_model_size(self):
        for residual in [False,True]:
            n=sum(p.numel() for p in ConditionalUNet(residual=residual).parameters())
            self.assertTrue(5_000_000<=n<=20_000_000)


if __name__=='__main__':unittest.main()
