import time,torch
from tensorboardX import SummaryWriter
import datetime
import os
def train_model(model,num_epochs,data,device,optimizer,criterion,datalen,savepath):

    nowtim = datetime.datetime.now().strftime('%m%d-%H%M')

    fl = nowtim +'CNN'

    writer = SummaryWriter(fl)

    since = time.time()

    best_acc =0.0

    _dic = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))

        for phase in ['train','test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for x1,y1 in data[phase]:
                x1 = torch.unsqueeze(x1, 1).float()
                x1 = x1.to(device)
                y1 = y1.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(x1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, y1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * x1.size(0)
                running_corrects += torch.sum(preds == y1.data)

            if phase == 'train':
                epoch_loss = running_loss / datalen[phase]
                epoch_acc = running_corrects.double() / datalen[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Acc', epoch_acc, epoch)
            else:
                epoch_loss = running_loss / datalen[phase]
                epoch_acc = running_corrects.double() / datalen[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('Test/Loss', epoch_loss, epoch)
                writer.add_scalar('Test/Acc', epoch_acc, epoch)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                    _dic.update(
                        {
                            'epoch': num_epochs,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best Acc': best_acc,

                        }
                    )

    print()
    writer.close()
    time_elapsed = time.time() - since

    filename = nowtim + 'acc' + str(int(best_acc * 100)) + '.t7'
    path = os.path.join(savepath, filename)
    torch.save(_dic,path)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model


