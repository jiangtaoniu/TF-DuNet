import argparse
import torch
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# Import other experiment types if you need them
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

parser = argparse.ArgumentParser(description='TF-DuNet: Time-Frequency Dual Network')

# =====================================================================================
# 1. Core Execution & Experiment Configuration
# =====================================================================================
parser.add_argument('--is_training', type=int, default=1, help='Status: 1 for training, 0 for testing')
parser.add_argument('--model_id', type=str, default='PEMS08', help='An identifier for the model experiment')
parser.add_argument('--model', type=str, default='TF_DuNet', help='Model name')
parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size of training input data')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate')
parser.add_argument('--lradj', type=str, default='TST', help='Learning rate adjustment strategy')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
parser.add_argument('--itr', type=int, default=1, help='Number of experiment iterations')
parser.add_argument('--des', type=str, default='Exp', help='A description for the experiment')
parser.add_argument('--comment', type=str, default='none', help='A comment for logging')

# -- Hardware Setup --
parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type [cuda, mps]')
parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='Device IDs for multiple GPUs')

# =====================================================================================
# 2. Data Loader & Input Configuration
# =====================================================================================
parser.add_argument('--data', type=str, default='PEMS', help='Dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/PEMS/', help='Root path of the data file')
parser.add_argument('--data_path', type=str, default='PEMS08_714.npz', help='Data file name')
parser.add_argument('--features', type=str, default='M', help='Forecasting task options: [M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='Target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t', help='Frequency for time features encoding')
parser.add_argument('--num_workers', type=int, default=10, help='Number of data loader workers')
parser.add_argument('--drop_last', type=bool, default=True, help='Whether to drop the last incomplete batch')

# =====================================================================================
# 3. Task & General Model Specification
# =====================================================================================
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name')
parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')
parser.add_argument('--label_len', type=int, default=0, help='Start token length')

# -- Core Dimensions & Layers --
parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
parser.add_argument('--d_ff', type=int, default=256, help='Dimension of the feed-forward network')
parser.add_argument('--enc_in', type=int, default=170, help='Encoder input size (number of features)')
parser.add_argument('--dec_in', type=int, default=170, help='Decoder input size')
parser.add_argument('--c_out', type=int, default=170, help='Output size')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--e_layers', type=int, default=5, help='Number of encoder layers for Decomposition Branch')
parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')

# -- General Modules & Activations --
parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding')
parser.add_argument('--channel_independence', type=int, default=0, help='0: channel dependence, 1: channel independence')
parser.add_argument('--use_norm', type=int, default=0, help='Whether to use normalization in the model')
parser.add_argument('--use_future_temporal_feature', type=int, default=0, help='Whether to use future temporal features (1 for True, 0 for False)')

# =====================================================================================
# 4. TF-DuNet Specific Parameters
# =====================================================================================
# -- Main Fusion Strategy --
parser.add_argument('--final_fusion_method', type=str, default='add',
                    choices=['add', 'static_weighted', 'concat', 'gate', 'film', 'cross_attention', 'learnable_weighted', 'custom_ranged'],
                    help='Final fusion method between ASST and Decomposition branches.')

# -- Decomposition Branch & MSDB Parameters --
parser.add_argument('--decomp_method', type=str, default='dft_decomp', help='Series decomposition method')
parser.add_argument('--top_k', type=int, default=5, help='Top-k frequencies for DFT decomposition')
parser.add_argument('--down_sampling_layers', type=int, default=1, help='Number of down-sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='Down-sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='Down-sampling method')
parser.add_argument('--msdb_internal_fusion_method', type=str, default='cross_attention',
                    choices=['add', 'static_weighted', 'concat', 'gate', 'film', 'cross_attention'],
                    help='Internal fusion method for MSDB within the decomposition branch.')
parser.add_argument('--node_dim', type=int, default=10, help='Dimension for node embeddings in MSDB')
parser.add_argument('--gcn_depth', type=int, default=2, help='Depth of graph propagation in MSDB')
parser.add_argument('--gcn_dropout', type=float, default=0.3, help='Dropout rate for GCN in MSDB')
parser.add_argument('--propalpha', type=float, default=0.3, help='Alpha for graph propagation in MSDB')
parser.add_argument('--conv_channel', type=int, default=32, help='Channel size for convolutions in MSDB')
parser.add_argument('--skip_channel', type=int, default=32, help='Channel size for skip connections in MSDB')
parser.add_argument('--gd_model', type=int, default=128, help='Dimension of graph-dependent model component')


# -- ASST (formerly STAEformer) Branch Parameters --
parser.add_argument('--asst_steps_per_day', type=int, default=288, help='ASST: number of samples per day')
# ================================== THE FIX IS HERE ==================================
parser.add_argument('--asst_input_dim', type=int, default=3, help='ASST: input feature dimension per node (e.g., 3 for value, tod, dow)')
# =====================================================================================
parser.add_argument('--asst_input_embedding_dim', type=int, default=24, help='ASST: input embedding dimension')
parser.add_argument('--asst_tod_embedding_dim', type=int, default=24, help='ASST: time of day embedding dimension')
parser.add_argument('--asst_dow_embedding_dim', type=int, default=24, help='ASST: day of week embedding dimension')
parser.add_argument('--asst_spatial_embedding_dim', type=int, default=0, help='ASST: spatial embedding dimension')
parser.add_argument('--asst_adaptive_embedding_dim', type=int, default=80, help='ASST: adaptive embedding dimension')
parser.add_argument('--asst_feed_forward_dim', type=int, default=256, help='ASST: feed forward dimension')
parser.add_argument('--asst_use_mixed_proj', type=bool, default=True, help='ASST: set to use mixed projection')
parser.add_argument('--asst_output_dim', type=int, default=1, help='ASST: output dimension of the branch')
parser.add_argument('--asst_n_heads', type=int, default=4, help='ASST: number of attention heads')
parser.add_argument('--asst_e_layers', type=int, default=3, help='ASST: number of encoder layers')

# =====================================================================================
# 5. Optimization & Training Strategy
# =====================================================================================
parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start for learning rate scheduler')
parser.add_argument('--inverse', action='store_true', help='Inverse output data for evaluation', default=False)
parser.add_argument('--use_dtw', action='store_true', default=False, help='Enable DTW for evaluation')

# =====================================================================================
# 6. Framework & Task-Specific Parameters
# =====================================================================================
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Seasonal patterns for M4 dataset, required by data provider')
parser.add_argument('--num_class', type=int, default=7, help='Number of classes for classification task')
parser.add_argument('--mask_rate', type=float, default=0.25, help='Mask rate for imputation task')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='Anomaly ratio for anomaly detection task')
parser.add_argument('--factor', type=int, default=3, help='Attention factor for some models like Informer')
parser.add_argument('--distil', type=bool, default=True, help='Use distilling in encoder for some models')
parser.add_argument('--output_attention', action='store_true', help='Whether to output attention in encoder')
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions of projector for some models')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='Number of hidden layers in projector for some models')
parser.add_argument('--moving_avg', type=int, default=25, help='Window size of moving average')
parser.add_argument('--num_kernels', type=int, default=6, help='Number of kernels for Inception')

# =====================================================================================
# Main Program Entry
# =====================================================================================
if __name__ == '__main__':
    args = parser.parse_args()

    fix_seed = 2021
    set_seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    Exp_classes = {
        'long_term_forecast': Exp_Long_Term_Forecast,
        'short_term_forecast': Exp_Short_Term_Forecast,
        'imputation': Exp_Imputation,
        'anomaly_detection': Exp_Anomaly_Detection,
        'classification': Exp_Classification
    }
    Exp = Exp_classes[args.task_name]

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_MSDBfus-{}_FinalFus-{}_sl{}_pl{}_dm{}_el{}_al{}_itr{}'.format(
                args.model,
                args.model_id,
                args.data,
                args.msdb_internal_fusion_method,
                args.final_fusion_method,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.asst_e_layers,
                ii
            )

            exp = Exp(args)
            print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_MSDBfus-{}_FinalFus-{}_sl{}_pl{}_dm{}_el{}_al{}_itr{}'.format(
                args.model,
                args.model_id,
                args.data,
                args.msdb_internal_fusion_method,
                args.final_fusion_method,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.asst_e_layers,
                ii
            )

        exp = Exp(args)
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()