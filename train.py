
from encoder_decoder import EncoderDecoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from refinement import RefineNet


device = 0
ed_epoch = 2
refine_epoch = 10
final_epoch = 100
batch_size = 16

RF = RefineNet().double().cuda(device)
ED = EncoderDecoder().double().cuda(device)

opt_ED = optim.Adam(ED.parameters(), lr=1e-4)
opt_RF = optim.Adam(RF.parameters(), lr=1e-4)

fg_path = '/data1/zhuyuanji01/Adobe_Deep_Image_Matting_Dataset/dim431/fg/'
bg_path = '/data1/zhuyuanji01/cocostuff/dataset/images/train2017/'
a_path = '/data1/zhuyuanji01/Adobe_Deep_Image_Matting_Dataset/dim431/alpha/'
merged_path = '/data1/zhuyuanji01/Adobe_Deep_Image_Matting_Dataset/dim431/merged/'

dataset = MattingDataSet(fg_path=fg_path, bg_path=bg_path, a_path=a_path, merged_path=merged_path)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':

    print('Beginning to PreTrain the Encoder Decoder')

    for epoch in range(ed_epoch):
        cnt = 0
        total_loss = 0
        for batch in dataloader:
            cnt += 1
            img, fg, bg, alpha, trimap, unknown = batch['img'].cuda(device), batch['fg'].cuda(device), batch['bg'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)

            input = torch.cat((img, trimap), 1)
            alpha_predict = ED(input)
            img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss_alpha = F.mse_loss(alpha_predict * unknown, alpha * unknown)
            loss = loss_comp + loss_alpha
            print(loss.item())
            total_loss += loss.item()
            opt_ED.zero_grad()
            loss.backward()
            opt_ED.step()
            if cnt % 100 == 0:
                print("epoch", epoch,cnt * 16 ,total_loss/100)
                total_loss = 0

    print('Beginning to PreTrain the RefineNet')

    for epoch in range(refine_epoch):
        for batch in dataloader:
            img, fg, bg, alpha, trimap, unknown = batch['img'].cuda(device), batch['fg'].cuda(device), batch['bg'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)
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
            img, fg, bg, alpha, trimap, unknown = batch['img'].cuda(device), batch['fg'].cuda(device), batch['bg'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)
            input = torch.cat((img, trimap), 1)
            alpha_raw = ED(input) * unknown + alpha * (1 - unknown)
            alpha_refined = RF(alpha_raw)
            loss_refine = F.mse_loss(alpha_refined, alpha)
            opt_RF.zero_grad()
            opt_ED.zero_grad()
            loss_refine.backward()
            opt_RF.step()
            opt_ED.step()













