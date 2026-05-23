import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
, mase_loss, smape_loss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
, accelerated_dtw
, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Experiment class encompassing the training, testing, and validation workflow
    specifically tailored for Long Term Spatio-Temporal Forecasting operations.

    This orchestrates the optimization cycle, logging, loss generation and manages
    the lifecycle of instantiated multi-scale graph convolutional transformer structures.
    """
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        # ================== Core Modification: Delayed initialization of experiment ID ==================
        # The experiment ID is no longer created in the constructor, initialized as None.
        # This allows the object to be instantiated successfully even with misconfigured args.
        self.setting_with_time = None
        # =================================================================

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS':
                    ### for stae cut data
                    batch_y = batch_y[:, :, :, 0].float().to(self.device)

                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred_np = pred.cpu().numpy()
                    true_np = true.cpu().numpy()
                    pred_np = vali_data.inverse_transform(pred_np.reshape(-1, C)).reshape(B, T, C)
                    true_np = vali_data.inverse_transform(true_np.reshape(-1, C)).reshape(B, T, C)
                    mae, _, _, _, _ = metric(pred_np, true_np)
                    total_loss.append(mae)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # ================== Core Modification: Receive setting and create a unique ID with timestamp ==================
        # This method now receives a base setting string and appends a timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setting_with_time = f"{timestamp}_{setting}"
        # ========================================================================

        print(f"Start training, experiment ID: {self.setting_with_time}")

        path = os.path.join(self.args.checkpoints, self.setting_with_time)
        if not os.path.exists(path):
            os.makedirs(path)

        log_file_path = os.path.join(path, 'run_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Experiment settings: {self.setting_with_time}\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log_file.write("------------------- Hyperparameters -------------------\n")
            for arg, value in sorted(vars(self.args).items()):
                log_file.write(f"{arg}: {value}\n")
            log_file.write("-----------------------------------------------------\n\n")
            log_file.write("-------------------- Training Log -------------------\n")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start, epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS':
                    ### for stae cut data
                    batch_y = batch_y[:, :, :, 0].float().to(self.device)


                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    # print('loss:{}'.format(loss))
                    # print(f"Outputs shape: {outputs.shape}")  # Print shape for reference
                    # print(f"Batch_y shape: {batch_y.shape}")  # Print shape for reference
                    # print("----------- Values -----------")
                    # # Use .detach().cpu().numpy() to print array, direct tensor is also fine
                    # print(f"Outputs value: \n{outputs}")
                    # print(f"Batch_y value: \n{batch_y}")
                    # # Use .item() to get scalar value of loss
                    # print(f"Loss value: {loss.item()}")

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    scheduler.step()

            current_epoch_time = time.time() - epoch_time
            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, current_epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            with open(log_file_path, 'a') as log_file:
                log_file.write(
                    f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.7f} | Val Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | Cost Time: {current_epoch_time:.2f}s\n"
                )
                if self.args.lradj == 'TST':
                    log_file.write(f"    -> Learning rate updated to: {scheduler.get_last_lr()[0]:.7f}\n")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"\nEarly stopping at epoch {epoch + 1}.\n")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Learning rate updated to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
        return self.model

    def test(self, setting, test=0):
        # Fix: The test method now receives a setting parameter, primarily for loading models not trained by this instance
        # For the train -> test workflow, it will use self.setting_with_time
        if self.setting_with_time is None:
            # If test is explicitly called independently, use the passed setting directly
            self.setting_with_time = setting

        print(f"Start testing, experiment ID: {self.setting_with_time}")

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('Loading model...')
            model_path = os.path.join('./checkpoints/', self.setting_with_time, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

        preds = []
        trues = []
        folder_path = './test_results/' + self.setting_with_time + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS':
                    ### for stae cut data
                    batch_y = batch_y[:, :, :, 0].float().to(self.device)

                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                # print(f"Shape Comparison -> batch_y: {batch_y.shape}, outputs: {outputs.shape}")
                # print()
                if i % 20 == 0:
                    if self.args.data in ['ETTh1','ETTh2','ETTm1','ETTm2','weather','electricity']:
                        input_data = batch_x.detach().cpu().numpy()
                    else:
                        input_data = batch_x[:, :, :, 0].detach().cpu().numpy()
                    # print(f"Shape Comparison -> input_data: {input_data[0, :, -1].shape}, batch_y: {batch_y[0, :, -1].shape}, outputs: {outputs[0, :, -1].shape}")

                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test set shape before inverse normalization:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
            print('Test set shape after inverse normalization:', preds.shape, trues.shape)
        elif test_data.scale and self.args.inverse:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("Calculate dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
        else:
            dtw_val = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('--- Final Test Metrics ---')
        print(f'MSE:  {mse:.7f}')
        print(f'MAE:  {mae:.7f}')
        print(f'RMSE: {rmse:.7f}')
        print(f'MAPE: {mape:.7f}')
        print(f'MSPE: {mspe:.7f}')
        print(f'DTW:  {dtw_val}')
        print('--------------------')

        log_file_path = os.path.join(self.args.checkpoints, self.setting_with_time, 'run_log.txt')
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write("\n--------------- Final Test Results ---------------\n")
                log_file.write(f"Tested at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f'MSE:  {mse:.7f}\n')
                log_file.write(f'MAE:  {mae:.7f}\n')
                log_file.write(f'RMSE: {rmse:.7f}\n')
                log_file.write(f'MAPE: {mape:.7f}\n')
                log_file.write(f'MSPE: {mspe:.7f}\n')
                log_file.write(f'DTW:  {dtw_val}\n')
                log_file.write("--------------------------------------------------\n")
        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file_path}. Skipping test metrics logging.")

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(self.setting_with_time + "  \n")
            f.write(f"mse:{mse:.7f}, mae:{mae:.7f}, rmse:{rmse:.7f}, mape:{mape:.7f}, mspe:{mspe:.7f}, dtw:{dtw_val}")
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
