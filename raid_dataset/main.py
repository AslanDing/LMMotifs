import os
import torch
from torch.optim.swa_utils import AveragedModel
from graph_build import build_text_graph_dataset1
from model import TextGNN
from eval import eval,MovingAverage

from pprint import pprint


def main():
    num_epochs = 10000
    device = 'cuda:5'
    validation_metric = 'accuracy'

    dataset_dir = '../data/raid/subset'
    textdataset =  build_text_graph_dataset1(dataset_dir,window_size=20)
    train_dataset, val_dataset, test_dataset = textdataset.tvt_split()

    train_dataset.init_node_feats('one_hot_init', device)
    val_dataset.init_node_feats('one_hot_init', device)


    moving_avg = MovingAverage(1000, validation_metric != 'loss')

    # build model
    model = TextGNN(pred_type='softmax', node_embd_type='gcn',
            num_layers=2, layer_dim_list=[55698,64,2], act='silu',
            bn=True, num_labels=2, class_weights=True, dropout=True)
    model.to(device)
    # decay = 0.999
    # model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    model = AveragedModel(model)
    model.load_state_dict(torch.load('../data/raid/logs/model.pth'))

    # train
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    pyg_graph = train_dataset.get_pyg_graph(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)

    best_results_val = 0
    best_results_test = 0
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        loss, preds_train = model(pyg_graph, train_dataset)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        model.update_parameters(model)
        with torch.no_grad():
            val_loss, preds_val = model(pyg_graph, val_dataset)
            test_loss, preds_test = model(pyg_graph, test_dataset)

            val_loss = val_loss.item()
            eval_res_val = eval(preds_val, val_dataset)
            print("Epoch: {:04d}, Train Loss: {:.5f}".format(epoch, loss))
            print("Val Loss: {:.5f}".format(val_loss))
            # print("Val Results: ...")
            # pprint(eval_res_val)

            eval_res_val["loss"] = val_loss


            eval_res_test = eval(preds_test, test_dataset)
            # print("test")
            # pprint(eval_res_test)

            moving_avg.add_to_moving_avg(eval_res_val[validation_metric])
            if best_results_val<=eval_res_val[validation_metric]:
                path = '../data/raid/logs/model.pth'
                torch.save(model.state_dict(), path)
                best_results_val = eval_res_val[validation_metric]
                best_results_test = eval_res_test[validation_metric]
            # if len(moving_avg.results) == 0 or moving_avg.best_result(eval_res_val[validation_metric]):
            #     path = '../data/raid/logs/model.pth'
            #     torch.save(model.state_dict(), path)

            # if moving_avg.stop():
            #     break
            print("current eval: ", eval_res_val[validation_metric])
            print("current test: ",eval_res_test[validation_metric])

            print("best eval: ",best_results_val)
            print("best test: ",best_results_test)

    # test results


if __name__=="__main__":
    main()