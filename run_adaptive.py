import torch
import csv
import os
from transformers import AutoModel
from src.svd_wrapper import count_parameters
from src.adaptive_compressor import apply_adaptive_svd

def main():
    device = torch.device("cpu")
    print("Loading baseline BERT model...")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    orig_params = count_parameters(model)
    
    energy_target = 0.90
    print(f"\nRunning Adaptive SVD Compression (Target Energy: {energy_target * 100}%)...")
    
    model, ranks_chosen = apply_adaptive_svd(model, energy_threshold=energy_target)
    
    final_params = count_parameters(model)
    reduction = ((orig_params - final_params) / orig_params) * 100
    
    print("=" * 70)
    print(f"Original Parameters:   {orig_params:,}")
    print(f"Compressed Parameters: {final_params:,}")
    print(f"Total Net Reduction:   {reduction:.2f}%")
    print("=" * 70)
    
    # Automatically save or append metrics to your benchmark spreadsheet
    csv_file = "svd_bert_benchmarks.csv"
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Energy_Target", "Original_Parameters", "Compressed_Parameters", "Net_Reduction_Pct"])
        writer.writerow(["bert-base-uncased", energy_target, orig_params, final_params, f"{reduction:.2f}"])
        
    print(f"Metrics successfully saved to '{csv_file}'!")

if __name__ == "__main__":
    main()