import unittest
import torch
from paired_model import PositionUNet
from train import predict


class PairTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_positions_are_fixed_and_oriented(self):
        model=PositionUNet(base=4)
        self.assertEqual(tuple(model.position.shape),(1,2,64,128))
        self.assertTrue(torch.equal(model.position[0,0,:,0],torch.linspace(-1,1,64)))
        self.assertTrue(torch.equal(model.position[0,1,0,:],torch.linspace(-1,1,128)))
        self.assertNotIn('position',dict(model.named_parameters()))

    def test_identical_initialization_and_input_planes(self):
        torch.manual_seed(6180);regression=PositionUNet(base=4)
        torch.manual_seed(6180);diffusion=PositionUNet(base=4)
        self.assertTrue(all(torch.equal(v,diffusion.state_dict()[k]) for k,v in regression.state_dict().items()))
        self.assertEqual(regression.initial_state_sha256,diffusion.initial_state_sha256)
        observed=[]
        hook=diffusion.input.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach()))
        noisy=torch.randn(2,2,64,128);diffusion(noisy,torch.tensor([1,900]),torch.randn(2,4));hook.remove()
        self.assertTrue(torch.equal(observed[0][:,:2],noisy))
        self.assertTrue(torch.equal(observed[0][:,2:],diffusion.position.expand(2,-1,-1,-1)))

    def test_deterministic_prediction_ignores_targets_and_clamps_mask(self):
        model=PositionUNet(base=4).eval();torch.nn.init.normal_(model.output[-1].weight,std=.01)
        data={'x':torch.randn(2,4),'u':torch.randn(2,2,64,128),'mask':torch.zeros(2,64,128,dtype=torch.bool)}
        data['mask'][:,30:33,60:65]=True;ids=torch.arange(2)
        mean=torch.zeros(1,2,1,1);std=torch.ones_like(mean)
        a=predict(model,data,ids,mean,std);data['u'].normal_(100,100);b=predict(model,data,ids,mean,std)
        self.assertTrue(torch.equal(a,b));self.assertTrue((a.masked_select(data['mask'][:,None])==0).all())
        self.assertLessEqual(float(a.detach().abs().max()),1)

    def test_full_backbone_parameter_count(self):
        model=PositionUNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()),5_602_322)


if __name__=='__main__':unittest.main()
