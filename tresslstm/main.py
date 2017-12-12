#-*- coding:utf-8 -*-

from train_model import Trainer,genDataset
import time
from test_pytorch_lstm import TreeLstm
import torch

input_size=100
hidden_size=200
batch_size=128
if __name__ == '__main__':

    model = TreeLstm(input_size,hidden_size)
    model.load_state_dict(torch.load("./train_status.pkl"))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    trainer = Trainer(model,criterion,optimizer,batch_size)
    train_dataset = genDataset('./phone_tree/phone_seged_sent_val','./phone_tree/phone_s_tree_val')
    val_dataset = genDataset('./phone_tree/phone_seged_sent_val','./phone_tree/phone_s_tree_val')
#    train_dataset = genDataset('./train_seged_sent','./train_s_tree')
#    val_dataset  = genDataset('./train_seged_sent','./train_s_tree')

    since=time.time()
    best_acc=0.0
    best_model_wts = model.state_dict()

    for epoch in range(20):
         train_loss             = trainer.train(train_dataset)
        #train_loss, train_accu = trainer.test(train_dataset)
        # test_loss, test_pred   = trainer.test(test_dataset)

        #print('---------------train---------------')
        #print('TrainLoss: {:.4f} ValAcc: {:.4f}'.format(
        #        train_loss, train_accu))
        print('--------------val------------------')
        val_loss ,val_accu    = trainer.test(val_dataset)
        print('ValLoss: {:.4f} ValAcc: {:.4f}'.format(
               val_loss, val_accu))

            # deep copy the model
        if val_accu > best_acc:
            best_acc = val_accu
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./train_status.pkl")
