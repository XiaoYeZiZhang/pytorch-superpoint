import torch
if __name__ == "__main__":
    model1 = torch.load("/home/zhangye/Develope/superpoint_pytorch/superpoint/logs/magicpoint_synth/checkpoints/superPointNet_4000_checkpoint.pth.tar", torch.device("cuda"))
    traced_script_module = torch.jit.trace(model1['model_state_dict'], torch.ones(1,1,640,480))
    traced_script_module.cuda()
    traced_script_module.save("/home/zhangye/data1/superpoint_v1_test2.pt")