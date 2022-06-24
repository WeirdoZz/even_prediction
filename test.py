import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset import SpecifiedDataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self,seq):
        super(MyDataset, self).__init__()
        self.seq=seq

    def __getitem__(self, index):
        return self.seq[index]

    def __len__(self):
        return self.seq.shape[0]
if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
    dataset=SpecifiedDataset.NYT()
    train_set, dev_set, test_set, num_events, kg_map = dataset.generatet_dataset()

    train_sequence=torch.tensor(train_set,requires_grad=False)
    dev_sequence=torch.tensor(dev_set,requires_grad=False)
    test_sequence=torch.tensor(test_set,requires_grad=False)

    enc=nn.Embedding(num_events,100)
    # train_seq=enc(train_sequence[:,:7])
    # train_tar = train_sequence[:, 7:9]
    # dev_seq=enc(dev_sequence[:,:7])
    # dev_tar = dev_sequence[:, 7:9]
    # test_seq=enc(test_sequence[:,:7])
    # test_tar = test_sequence[:, 7:9]

    train_set=MyDataset(train_sequence)
    dev_set=MyDataset(dev_sequence)
    test_set=MyDataset(test_sequence)

    train_loader=DataLoader(train_set,batch_size=1024,shuffle=True,num_workers=4)
    dev_loader=DataLoader(dev_set,batch_size=1024,num_workers=4)
    test_loader=DataLoader(test_set,batch_size=1024,num_workers=4)

    rnn=nn.LSTM(100,200,batch_first=True)
    linear=nn.Linear(200,200)
    fc=nn.Linear(200,num_events)
    sm=nn.Softmax(-1)
    optimizer=torch.optim.SGD([{"params":rnn.parameters(),"lr":1e-3},
                                {"params":linear.parameters(),"lr":1e-3},
                                {"params":fc.parameters(),"lr":1e-3},
                               {"params":enc.parameters(),"lr":1e-3}])
    criterion=nn.CrossEntropyLoss()

    max_hit10=0
    step=0
    max_epoch=0
    for epoch in range(200):
        for sequence in tqdm(train_loader):
            X=enc(sequence[:,:8])
            y = sequence[:, 8]

            _,(out,c_0)=rnn(X)
            out_distrbution=linear(out)
            out_distrbution=fc(out_distrbution)
            out_distrbution=sm(out_distrbution)
            loss=criterion(out_distrbution.squeeze(),y[:])

            # _,(out,c_0)=rnn(tar[:0],(out,c_0))
            # out_distrbution = linear(out)
            # out_distrbution = fc(out_distrbution)
            # out_distrbution = sm(out_distrbution)
            # loss += criterion(out_distrbution, y[:, 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            hit10=0
            all=0
            for sequence in tqdm(dev_loader):
                X=enc(sequence[:, :8])
                y = sequence[:, 8]

                _, (out, c_0) = rnn(X)
                out_distrbution = linear(out)
                out_distrbution = fc(out_distrbution)
                out_distrbution = sm(out_distrbution)
                out_distrbution=torch.argsort(out_distrbution.squeeze(),dim=-1,descending=True)
                for i in range(out_distrbution.shape[0]):
                    if y[i] in out_distrbution[i][:10]:
                        hit10+=1
                    all+=1

            hit10 = hit10 / all
            print(f"epoch:{epoch},hit10:{hit10}")
            if hit10>max_hit10:
                max_hit10=hit10
                step=0
                max_epoch=epoch
            else:
                step+=1
                if step>=20:
                    break
    print(f"最优周期：{max_epoch},hit10:{max_hit10}")












