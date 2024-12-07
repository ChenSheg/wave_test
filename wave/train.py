import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#script_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.dirname(script_dir)
#sys.path.append(project_root)

# 获取当前脚本的目录和父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 确保父目录在 sys.path 中
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#print(sys.path)

import logging
from timeit import default_timer as timer
import pandas as pd
import torch
from torch.utils.data import DataLoader
import scanpy as sc
from wave.utils import time_postfix, seed_everything, loss_fct, log_args
from wave.load_dataset import PerturbationDataset
from wave.inference import evaluate, compute_mean_metrics, compute_metrics
from wave.model import WAVE

#script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 模型目录

# 默认路径解析为脚本目录下的 `models/genevae.pth`
default_genevae_model_path = os.path.join(model_dir, "models", "genevae.pth")

def train_wave(args):
    """
    Train the WAVE model and log metrics per epoch to a CSV file.
    """
    # 设置随机种子
    seed_everything(getattr(args, "seed", 1))

    

    # Convert relative path to absolute path
    #args.genevae_model_path = os.path.abspath(args.genevae_model_path)
    args.genevae_model_path = os.path.abspath(getattr(args, "genevae_model_path", default_genevae_model_path))

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    checkpoint_dir = os.path.join(args.outdir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 日志设置
    logging.basicConfig(filename=f"{args.outdir}/training{time_postfix()}.log", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    log_args(args, logger)

    

    logger.info(f"Start Training.")
    print("Start Training.")
    

    # 加载数据
    train_dataset = PerturbationDataset(args.train_dataset)
    val_dataset = PerturbationDataset(args.val_dataset)
    test_dataset = PerturbationDataset(args.test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    #model = WAVE()
    
    """model = WAVE(
        genevae_model_path=args.genevae_model_path,
        gene_input_dim=args.gene_input_dim,
        drug_input_dim=args.drug_input_dim,
        fusion_dim=args.fusion_dim,
        gene_output_dim=args.gene_output_dim,
        drug_hidden_dim=args.drug_hidden_dim,
        drug_output_dim=args.drug_output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(args.device)"""

    # 初始化模型
    model = WAVE(
        genevae_model_path=args.genevae_model_path,
        gene_input_dim=getattr(args, "gene_input_dim", 978),
        drug_input_dim=getattr(args, "drug_input_dim", 2048),
        fusion_dim=getattr(args, "fusion_dim", 512),
        gene_output_dim=getattr(args, "gene_output_dim", 978),
        drug_hidden_dim=getattr(args, "drug_hidden_dim", 256),
        drug_output_dim=getattr(args, "drug_output_dim", 2048),
        num_heads=getattr(args, "num_heads", 8),
        num_layers=getattr(args, "num_layers", 2)
        ).to(args.device)
    
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_log2fc_pearson = float("-inf")
    best_model_path = os.path.join(args.outdir, "best_model.pth")
    best_epoch = 0

    # 创建存储 epoch 记录的 DataFrame
    metrics_log = []

    # 训练循环
    for epoch in range(args.max_epochs):
        start_time = timer()

        # 训练阶段
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch["unpert_expr"] = batch["unpert_expr"].float().to(args.device)
            batch["pert_expr"] = batch["pert_expr"].float().to(args.device)
            batch['drug_fp'] = batch['drug_fp'].float().to(args.device)

            predictions = model(batch)
            loss = loss_fct(predictions, batch["pert_expr"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        mean_train_loss = epoch_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch["unpert_expr"] = batch["unpert_expr"].float().to(args.device)
                batch["pert_expr"] = batch["pert_expr"].float().to(args.device)
                batch['drug_fp'] = batch['drug_fp'].float().to(args.device)

                predictions = model(batch)
                val_loss += loss_fct(predictions, batch["pert_expr"]).item()
        mean_val_loss = val_loss / len(val_loader)

        val_results = evaluate(val_loader, model)
        val_mean_metrics = compute_mean_metrics(val_results)

        # 添加 epoch 记录
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "mean_val_mse": val_mean_metrics["mean_mse"],
            "mean_val_r2": val_mean_metrics["mean_r2"],
            "mean_val_pearson": val_mean_metrics["mean_pearson"],
            "mean_val_expr_change_pearson": val_mean_metrics["mean_expr_change_pearson"],
            "mean_val_log2fc_pearson": val_mean_metrics["mean_log2fc_pearson"]
        })

        end_time = timer()

        # 更新最优模型
        if val_mean_metrics["mean_log2fc_pearson"] > best_log2fc_pearson or epoch == 0:
            best_log2fc_pearson = val_mean_metrics["mean_log2fc_pearson"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

        

        log_msg = (f"Epoch {epoch + 1}/{args.max_epochs}: Train Loss = {mean_train_loss:.5f}, "
                   f"Val Loss = {mean_val_loss:.5f}, "
                   f"Mean Val MSE = {val_mean_metrics['mean_mse']:.5f}, "
                   f"Mean Val R2 = {val_mean_metrics['mean_r2']:.5f}, "
                   f"Mean Val Pearson = {val_mean_metrics['mean_pearson']:.5f}, "
                   f"Mean Expression Change Pearson = {val_mean_metrics['mean_expr_change_pearson']:.5f}, "
                   f"Mean Log2(FC) Pearson = {val_mean_metrics['mean_log2fc_pearson']:.5f}, "
                   f"Best Log2(FC) Pearson = {best_log2fc_pearson:.5f} at Epoch {best_epoch}, "
                   f"Epoch Time = {(end_time - start_time) / 3600:.3f}h")
        logger.info(log_msg)
        print(log_msg)

    logger.info(f"Training compeleted.")
    print("Training compeleted.")
    
    # 保存 metrics_log 到文件
    metrics_df = pd.DataFrame(metrics_log)
    metrics_file = os.path.join(args.outdir, "metrics_per_epoch.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    logger.info(f"Train/Val metrics saved to {metrics_file}")
    print(f"Train/Val metrics saved to {metrics_file}")

    # 测试阶段
    logger.info("Starting testing phase")
    print("Starting testing phase")

    model.load_state_dict(torch.load(best_model_path))
    test_results = evaluate(test_loader, model)
    test_mean_metrics = compute_mean_metrics(test_results)
    
    test_adata = sc.read_h5ad(args.test_dataset)
    test_adata.uns['mean_metrics'] = test_mean_metrics
    test_metrics = compute_metrics(test_results)
    test_metrics_df = pd.DataFrame(test_metrics)
    test_adata.uns['metrics'] = test_metrics_df
    #print(test_metrics.keys())
    test_adata.layers['prediction'] = test_results['prediction']
    test_adata.write_h5ad(os.path.join(args.outdir, "test_prediction.h5ad"))
    
    logger.info("Testing completed. Metrics saved.")
    print("Testing completed. Metrics saved.")

    # 清理 checkpoint
    if getattr(args, "clear_checkpoint", True):
        import shutil
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Cleared checkpoint directory: {checkpoint_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the WAVE model via CLI",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 通用参数
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dataset", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument('--clear_checkpoint', type=bool, default=True, help="Whether to clear the checkpoint folder after training")
    
    # 模型相关参数
    parser.add_argument("--genevae_model_path", type=str, default=default_genevae_model_path, help="Path to Gene VAE model")
    parser.add_argument("--gene_input_dim", type=int, default=978, help="Dimension of input gene expression")
    parser.add_argument("--drug_input_dim", type=int, default=2048, help="Dimension of the drug Morgan fingerprint")
    parser.add_argument("--fusion_dim", type=int, default=512, help="Dimension for gene and drug fusion")
    parser.add_argument("--gene_output_dim", type=int, default=978, help="Dimension of gene output")
    parser.add_argument("--drug_hidden_dim", type=int, default=256, help="Hidden dimension for drug fingerprint processing")
    parser.add_argument("--drug_output_dim", type=int, default=2048, help="Output dimension of drug embedding")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in Transformer Decoder")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer Decoder layers")

    args = parser.parse_args()

    train_wave(args)


if __name__ == "__main__":
    main()
