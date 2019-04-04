
from encoder_decoder import EncoderDecoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from refinement import RefineNet


device = 1
ed_epoch = 100
refine_epoch = 100
final_epoch = 100
batch_size = 128


RF = RefineNet().double().to(device)
ED = EncoderDecoder().double().to(device)

opt_ED = optim.RMSprop(ED.parameters())
opt_RF = optim.RMSprop(RF.parameters())
root = '/data/zhuzhanxing/Adobe_Deep_Image_Matting_Dataset/dim431'

fg_path = os.path.join(root, 'fg/')
bg_path = os.path.join(root, 'bg/')
a_path = os.path.join(root, 'alpha/')
merged_path = os.path.join(root, 'merged/merged/')

dataset = MattingDataSet(fg_path=fg_path, bg_path=bg_path, a_path=a_path, merged_path=merged_path)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size)



print('Beginning to PreTrain the Encoder Decoder')

for epoch in range(ed_epoch):
    for batch in dataloader:
        img, fg, bg, alpha, trimap, unknown = batch['img'].to(device), batch['fg'].to(device), batch['bg'].to(device), \
                                              batch['alpha'].to(device), batch['trimap'].to(device), \
                                              batch['unknown'].to(device)

        input = torch.cat((img, trimap), 1)
        alpha_predict = ED(input)
        img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
        loss_comp = F.l1_loss(img_predict * unknown, img * unknown)
        loss_alpha = F.mse_loss(alpha_predict * unknown, alpha * unknown)
        loss = loss_comp + loss_alpha
        print(loss)
        opt_ED.zero_grad()
        loss.backward()
        opt_ED.step()

print('Beginning to PreTrain the RefineNet')

for epoch in range(refine_epoch):
    for batch in dataloader:
        img, fg, bg, alpha, trimap, unknown = batch['img'].to(device), batch['fg'].to(device), batch['bg'].to(device), \
                                              batch['alpha'].to(device), batch['trimap'].to(device), \
                                              batch['unknown'].to(device)
        input = torch.cat((img, trimap), 1)
        alpha_raw = ED(input) * unknown + alpha * (1-unknown)
        alpha_refined = RF(alpha_raw)
        loss_refine = F.mse_loss(alpha_refined, alpha)
        opt_RF.zero_grad()
        loss_refine.backward()
        opt_RF.step()


print('Begining to Train the whole Model')

for epoch in range(final_epoch):
    for batch in dataloader:
        img, fg, bg, alpha, trimap, unknown = batch['img'].to(device), batch['fg'].to(device), batch['bg'].to(device), \
                                              batch['alpha'].to(device), batch['trimap'].to(device), \
                                              batch['unknown'].to(device)
        input = torch.cat((img, trimap), 1)
        alpha_raw = ED(input) * unknown + alpha * (1 - unknown)
        alpha_refined = RF(alpha_raw)
        loss_refine = F.mse_loss(alpha_refined, alpha)
        opt_RF.zero_grad()
        opt_ED.zero_grad()
        loss_refine.backward()
        opt_RF.step()
        opt_ED.step()













