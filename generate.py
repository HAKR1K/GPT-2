import os
import argparse
import torch
from tokenizer import SentencePieceTokenizer
from model import MiniGPT

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_prefix", type=str, default="data/processed/sp", help="Prefix for SentencePiece model file")
    parser.add_argument("--prompt", type=str, default="To be, or not to be, that is the question:", help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=150, help="Number of tokens to generate")
    parser.add_argument("--method", type=str, default="top_k", choices=["greedy", "temperature", "top_k", "top_p"], help="Sampling method")
    parser.add_argument("--temperature", type=float, default=0.8, help="Logits temperature scaling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering constraint")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p filtering constraint")
    parser.add_argument("--out_file", type=str, default="outputs/generated.txt", help="File path to save the generated output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = SentencePieceTokenizer()
    sp_model_file = f"{args.tokenizer_prefix}.model"
    if not os.path.exists(sp_model_file):
        raise FileNotFoundError(f"Tokenizer model not found at {sp_model_file}. Run train.py first to train it.")
    tokenizer.load(sp_model_file)

    if args.checkpoint == "checkpoints/checkpoint_best.pt" and not os.path.exists(args.checkpoint):
        part1 = "checkpoints/checkpoint_best.zip.aa"
        part2 = "checkpoints/checkpoint_best.zip.ab"
        if os.path.exists(part1) and os.path.exists(part2):
            import zipfile
            print("Reassembling model checkpoint from split zip files...")
            os.makedirs("checkpoints", exist_ok=True)
            temp_zip = "checkpoints/temp_checkpoint_best.zip"
            try:
                with open(temp_zip, "wb") as f_out:
                    for part in [part1, part2]:
                        with open(part, "rb") as f_in:
                            f_out.write(f_in.read())
                with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                    zip_ref.extractall(".")
                print("Model checkpoint successfully reassembled.")
            except Exception as e:
                print(f"Error reassembling checkpoint: {e}")
            finally:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")
    
    print(f"Loading model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model_config = checkpoint['config']
    model = MiniGPT(model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = model.generate(
        idx=x,
        max_new_tokens=args.max_tokens,
        method=args.method,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    output_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated Output:")
    print(output_text)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nSaved generated output to {args.out_file}")

if __name__ == "__main__":
    main()
