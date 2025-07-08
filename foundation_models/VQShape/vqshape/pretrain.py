import torch
import argparse
import sys, os
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb

from data_provider.timeseries_loader import TimeSeriesDatasetLazy
from vqshape.vqshape_utils import visualize_shapes, plot_code_heatmap
from vqshape.model import VQShape

import warnings
warnings.filterwarnings('ignore')


strategy_dict = {
    'auto': 'auto',
    'ddp': 'ddp',
    'fsdp': 'fsdp',
    'deepspeed': 'deepspeed_stage_2',
    'dp': 'dp'
}


class LitVQShape(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)
        try:
            self.model = VQShape(
                len_input=self.hparams.normalize_length,
                dim_embedding=self.hparams.dim_embedding,
                patch_size=self.hparams.patch_size,
                num_patch=self.hparams.num_patch,
                num_token=self.hparams.num_token,
                num_enc_head=self.hparams.num_transformer_enc_heads,
                num_enc_layer=self.hparams.num_transformer_enc_layers,
                num_tokenizer_head=self.hparams.num_tokenizer_heads,
                num_tokenizer_layer=self.hparams.num_tokenizer_layers,
                num_dec_head=self.hparams.num_transformer_dec_heads,
                num_dec_layer=self.hparams.num_transformer_dec_layers,
                len_s=self.hparams.len_s,
                s_smooth_factor=self.hparams.s_smooth_factor,
                num_code=self.hparams.num_code,
                dim_code=self.hparams.dim_code,
                codebook_type=self.hparams.codebook_type,
                lambda_commit=self.hparams.lambda_vq_commit,
                lambda_entropy=self.hparams.lambda_vq_entropy,
                entropy_gamma=self.hparams.entropy_gamma,
                mask_ratio=self.hparams.mask_ratio
            )
        except:
            warnings.warn("Incompatible configs. Trying the legacy version...")
            self.model = VQShape(
                len_input=self.hparams.normalize_length,
                dim_embedding=self.hparams.dim_embedding,
                patch_size=self.hparams.patch_size,
                num_patch=self.hparams.num_patch,
                num_token=self.hparams.num_token,
                num_enc_head=self.hparams.num_transformer_enc_heads,
                num_enc_layer=self.hparams.num_transformer_enc_layers,
                num_tokenizer_head=self.hparams.num_transformer_enc_heads,
                num_tokenizer_layer=self.hparams.num_transformer_enc_layers,
                num_dec_head=self.hparams.num_transformer_dec_heads,
                num_dec_layer=self.hparams.num_transformer_dec_layers,
                len_s=self.hparams.len_s,
                s_smooth_factor=self.hparams.s_smooth_factor,
                num_code=self.hparams.num_code,
                dim_code=self.hparams.dim_code,
                codebook_type='standard',
                lambda_commit=self.hparams.lambda_vq_commit,
                lambda_entropy=self.hparams.lambda_vq_entropy,
                entropy_gamma=self.hparams.entropy_gamma,
                mask_ratio=self.hparams.mask_ratio
            )

        self.validation_step_outputs = {}
        for i in range(1):
            self.validation_step_outputs[i] = {'z': [], 't': [], 'l': []}

    def training_step(self, batch, batch_idx):
        _, loss_dict = self.model(batch, mode='pretrain')

        loss = self.hparams.lambda_x * loss_dict['ts_loss'].mean() + \
            self.hparams.lambda_z * loss_dict['vq_loss'].mean() + \
            self.hparams.lambda_s * loss_dict['shape_loss'].mean() + \
            self.hparams.lambda_dist * loss_dict['dist_loss'].mean()
        
        loss_dict = {
            "TRAIN/total": loss, "TRAIN/x": loss_dict['ts_loss'].mean(), "TRAIN/z": loss_dict['vq_loss'].mean(), 
            "TRAIN/s": loss_dict['shape_loss'].mean(), 'TRAIN/dist': loss_dict['dist_loss'].mean()
        }
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict, loss_dict = self.model(batch, mode='evaluate')

        loss = self.hparams.lambda_x * loss_dict['ts_loss'].mean() +\
            self.hparams.lambda_z * loss_dict['vq_loss'].mean() + \
            self.hparams.lambda_s * loss_dict['shape_loss'].mean() + \
            self.hparams.lambda_dist * loss_dict['dist_loss'].mean()
        
        loss_dict = {
            f"VAL{dataloader_idx}/total": loss, 
            f"VAL{dataloader_idx}/x": loss_dict['ts_loss'].mean(), 
            f"VAL{dataloader_idx}/z": loss_dict['vq_loss'].mean(),
            f"VAL{dataloader_idx}/s": loss_dict['shape_loss'].mean(), 
            f'VAL{dataloader_idx}/dist': loss_dict['dist_loss'].mean()
        }
        self.log_dict(loss_dict, sync_dist=True)
        self.validation_step_outputs[dataloader_idx]['z'].append(output_dict['code_idx'])
        self.validation_step_outputs[dataloader_idx]['t'].append(output_dict['t_pred'])
        self.validation_step_outputs[dataloader_idx]['l'].append(output_dict['l_pred'])
        if batch_idx == 0 and self.global_rank == 0:
            fig, s_fig = visualize_shapes(output_dict)
            self.logger.experiment.log({
                f"VAL{dataloader_idx}/x_fig": wandb.Image(fig),
                f"VAL{dataloader_idx}/s_fig": wandb.Image(s_fig)
            }, self.global_step)
            plt.close(fig)
            plt.close(s_fig)

    def on_validation_epoch_end(self):
        for i in self.validation_step_outputs.keys():
            z_idx = torch.cat(self.validation_step_outputs[i]['z'], dim=0)
            t_hat = torch.cat(self.validation_step_outputs[i]['t'], dim=0)
            l_hat = torch.cat(self.validation_step_outputs[i]['l'], dim=0)
            fig = plot_code_heatmap(z_idx, self.hparams.num_code, title=f"{self.global_step}")
            self.logger.experiment.log({
                f"VAL{i}/z_dist": wandb.Image(fig),
                f"VAL{i}/t_dist": wandb.Histogram(t_hat.float().cpu().numpy()),
                f"VAL{i}/l_dist": wandb.Histogram(l_hat.float().cpu().numpy())
            }, self.global_step)
            for key in self.validation_step_outputs[i].keys():
                self.validation_step_outputs[i][key].clear()
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        stepping_batches = self.trainer.estimated_stepping_batches - self.hparams.warmup_step

        # Cosine annealing with linear warmup learning rate schedule
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, 0.0001, 1, self.hparams.warmup_step), 
                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stepping_batches, eta_min=1e-5)], 
            milestones=[self.hparams.warmup_step]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    

def get_args():
    parser = argparse.ArgumentParser()

    # Model configs
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--normalize_length", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_patch", type=int, default=64)
    parser.add_argument("--num_token", type=int, default=32)
    parser.add_argument("--num_transformer_enc_heads", type=int, default=3)
    parser.add_argument("--num_transformer_enc_layers", type=int, default=6)
    parser.add_argument("--num_tokenizer_heads", type=int, default=3)
    parser.add_argument("--num_tokenizer_layers", type=int, default=6)
    parser.add_argument("--num_transformer_dec_heads", type=int, default=3)
    parser.add_argument("--num_transformer_dec_layers", type=int, default=6)
    parser.add_argument("--num_code", type=int, default=512)
    parser.add_argument("--dim_code", type=int, default=32)
    parser.add_argument("--codebook_type", type=str, default='standard', choices=['standard', 'vqtorch'])
    parser.add_argument("--len_s", type=int, default=256)
    parser.add_argument("--s_smooth_factor", type=int, default=11)
    parser.add_argument("--lambda_x", type=float, default=1.)
    parser.add_argument("--lambda_z", type=float, default=1.)
    parser.add_argument("--lambda_s", type=float, default=1.)
    parser.add_argument("--lambda_dist", type=float, default=1.)
    parser.add_argument("--lambda_vq_commit", type=float, default=1.)
    parser.add_argument("--lambda_vq_entropy", type=float, default=1.)
    parser.add_argument("--entropy_gamma", type=float, default=1.)

    # Training parameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_step", type=int, default=1000)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--train_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--balance_datasets", action='store_true')

    # Environment settings
    parser.add_argument("--data_root", type=str, default='../data/VQShape')
    parser.add_argument("--dev", action='store_true')
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='auto')
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--val_frequency", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    args.save_dir = f"./checkpoints/vqshape_pretrain/{args.name}"
    return args


def main(args):

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    torch.set_float32_matmul_precision('medium')
    logger = WandbLogger(save_dir=args.save_dir, name=f"pretrain-{args.name}", project='VQShape', log_model=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        fast_dev_run=args.dev,
        default_root_dir=args.save_dir,
        strategy=strategy_dict[args.strategy],
        accelerator='gpu',
        devices=args.num_devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        log_every_n_steps=100,
        max_epochs=args.train_epoch,
        val_check_interval=args.val_frequency,
        limit_val_batches=1000,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[lr_monitor]
    )

    print(f"Start Time: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
    sys.stdout.flush()
    
    train_dataset = TimeSeriesDatasetLazy(
        args.data_root, 
        tasks=['uea'],
        split='TRAIN', 
        sequence_length=args.normalize_length, 
        balance=args.balance_datasets
    )
    val_dataset = TimeSeriesDatasetLazy(
        args.data_root, 
        tasks=['uea'],
        split='TEST', 
        sequence_length=args.normalize_length
    )

    print(f"Datasets: TRAIN {len(train_dataset)} | VAL {len(val_dataset)}")
    sys.stdout.flush()

    pin_memory = True
    shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory, drop_last=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory, drop_last=False, num_workers=args.num_workers)

    with trainer.init_module():
        model = LitVQShape(args)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # val_dataloaders=[val_loader, test_loader]
        val_dataloaders=val_loader
    )

    trainer.save_checkpoint(f"{args.save_dir}/VQShape.ckpt")


if __name__ == '__main__':
    args = get_args()
    main(args)